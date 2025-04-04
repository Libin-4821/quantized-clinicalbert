import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import os
import time
import sys
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.data_loader import get_dataloaders, save_sample_data
from src.training.model import ClinicalBERTClassifier, save_model, quantize_model

def train_model(data_file, 
                model_name="emilyalsentzer/Bio_ClinicalBERT",
                output_dir="models/clinicalbert",
                batch_size=16,
                epochs=5,
                learning_rate=2e-5,
                warmup_steps=0,
                max_length=512,
                device=None,
                quantize=False):
    """
    Train the ClinicalBERT model for disease classification
    
    Args:
        data_file (str): Path to CSV file with symptoms and disease labels
        model_name (str): Name of the pre-trained model
        output_dir (str): Directory to save model
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        warmup_steps (int): Number of warmup steps for scheduler
        max_length (int): Maximum sequence length for tokenizer
        device (str): Device to train on (cpu or cuda)
        quantize (bool): Whether to quantize the model to 4-bit after training
        
    Returns:
        tuple: (trained model, tokenizer, disease_to_id, id_to_disease)
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Training on {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader, disease_to_id, id_to_disease = get_dataloaders(
        data_file=data_file,
        tokenizer_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )
    
    num_classes = len(disease_to_id)
    print(f"Found {num_classes} disease classes: {list(disease_to_id.keys())}")
    
    # Initialize model
    model = ClinicalBERTClassifier(model_name=model_name, num_classes=num_classes)
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Define scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track progress
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(true_labels)
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1 Score: {train_f1:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Track progress
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = labels.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(true_labels)
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                disease_to_id=disease_to_id,
                id_to_disease=id_to_disease
            )
            print(f"Best model saved with F1 Score: {val_f1:.4f}")
    
    # Test best model
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))
    model.eval()
    
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Track progress
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(true_labels)
    
    # Calculate test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_report = classification_report(test_labels, test_preds, 
                                      target_names=list(disease_to_id.keys()))
    
    print(f"Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(test_report)
    
    # Quantize model if requested
    if quantize:
        quantized_output_dir = output_dir + "-4bit-quantized"
        quantize_model(output_dir, quantized_output_dir)
    
    return model, tokenizer, disease_to_id, id_to_disease

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ClinicalBERT for disease classification')
    
    parser.add_argument('--data_file', type=str, default='data/raw/symptoms_data.csv',
                        help='Path to CSV file with symptoms and disease labels')
    parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT',
                        help='Name of the pre-trained model')
    parser.add_argument('--output_dir', type=str, default='models/clinicalbert',
                        help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenizer')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cpu or cuda)')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize model to 4-bit after training')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample data before training')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        save_sample_data()
        print(f"Sample data created at {args.data_file}")
    
    # Train model
    train_model(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        quantize=args.quantize
    )
