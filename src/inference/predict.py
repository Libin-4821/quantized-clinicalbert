import torch
import os
import sys
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
from src.training.model import load_model

class DiseasePredictor:
    """Class for making disease predictions from symptoms"""
    
    def __init__(self, model_dir="models/clinicalbert-4bit-quantized", device=None):
        """
        Initialize the disease predictor
        
        Args:
            model_dir (str): Directory containing the model
            device (str): Device to run inference on (cpu or cuda)
        """
        self.model_dir = model_dir
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.load_model()
    
    def load_model(self):
        """Load the model, tokenizer, and class mappings"""
        try:
            # Get number of classes from class mappings
            mappings = torch.load(os.path.join(self.model_dir, "class_mappings.pt"))
            num_classes = len(mappings['disease_to_id'])
            
            # Load model
            self.model, self.tokenizer, self.disease_to_id, self.id_to_disease = load_model(
                model_dir=self.model_dir,
                num_classes=num_classes
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully with {num_classes} disease classes")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, symptoms, top_k=3, threshold=0.1):
        """
        Predict diseases from symptoms
        
        Args:
            symptoms (str): Patient symptoms
            top_k (int): Number of top predictions to return
            threshold (float): Minimum probability threshold
            
        Returns:
            list: List of (disease, probability) tuples
        """
        # Tokenize symptoms
        inputs = self.tokenizer(
            symptoms,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], 
                               attention_mask=inputs["attention_mask"])
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        # Get top-k predictions above threshold
        indices = np.argsort(probs)[::-1]
        predictions = []
        
        for idx in indices:
            if len(predictions) >= top_k:
                break
                
            if probs[idx] >= threshold:
                disease = self.id_to_disease[idx]
                probability = float(probs[idx])
                predictions.append((disease, probability))
        
        return predictions
    
    def get_disease_info(self):
        """Get information about the diseases the model can predict"""
        disease_info = {}
        
        # Sample disease descriptions (to be expanded with actual info)
        descriptions = {
            "Heart_Failure": "A condition where the heart can't pump enough blood to meet the body's needs.",
            "Pneumonia": "An infection that inflames the air sacs in one or both lungs.",
            "Lung_Cancer": "A type of cancer that begins in the lungs and most often occurs in people who smoke.",
            "COPD": "Chronic Obstructive Pulmonary Disease, a chronic inflammatory lung disease.",
            "Tuberculosis": "A bacterial infection that primarily affects the lungs."
        }
        
        for disease_id, disease_name in self.id_to_disease.items():
            disease_info[disease_name] = {
                "id": disease_id,
                "description": descriptions.get(disease_name, "No description available")
            }
            
        return disease_info

def load_model_and_tokenizer(model_dir="models/clinicalbert-4bit-quantized"):
    """
    Helper function to load model and tokenizer for web app
    
    Args:
        model_dir (str): Directory containing the model
        
    Returns:
        tuple: (model, tokenizer, disease_to_id, id_to_disease)
    """
    predictor = DiseasePredictor(model_dir=model_dir)
    return predictor.model, predictor.tokenizer, predictor.disease_to_id, predictor.id_to_disease

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict disease from symptoms')
    parser.add_argument('--model_dir', type=str, default='models/clinicalbert-4bit-quantized',
                        help='Directory containing the model')
    parser.add_argument('--symptoms', type=str, required=True,
                        help='Patient symptoms')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to return')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DiseasePredictor(model_dir=args.model_dir)
    
    # Make prediction
    predictions = predictor.predict(args.symptoms, top_k=args.top_k)
    
    # Print predictions
    print(f"\nSymptoms: {args.symptoms}\n")
    print("Predictions:")
    for disease, prob in predictions:
        print(f"  {disease}: {prob:.4f} ({prob*100:.1f}%)")
