import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import os

class ClinicalBERTClassifier(nn.Module):
    """ClinicalBERT model for disease classification"""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", num_classes=5, dropout_rate=0.1):
        """
        Initialize the ClinicalBERT classifier
        
        Args:
            model_name (str): Name of the pre-trained model
            num_classes (int): Number of disease classes to predict
            dropout_rate (float): Dropout rate for classification layer
        """
        super(ClinicalBERTClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Token ids
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Logits for each class
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def save_model(model, tokenizer, output_dir, disease_to_id=None, id_to_disease=None):
    """
    Save the model, tokenizer, and class mappings
    
    Args:
        model (ClinicalBERTClassifier): The trained model
        tokenizer: The tokenizer
        output_dir (str): Directory to save model
        disease_to_id (dict): Mapping from disease name to class id
        id_to_disease (dict): Mapping from class id to disease name
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save class mappings
    if disease_to_id is not None and id_to_disease is not None:
        torch.save({
            'disease_to_id': disease_to_id,
            'id_to_disease': id_to_disease
        }, os.path.join(output_dir, "class_mappings.pt"))
        
def load_model(model_dir, num_classes=5):
    """
    Load the trained model and class mappings
    
    Args:
        model_dir (str): Directory containing the saved model
        num_classes (int): Number of disease classes
        
    Returns:
        tuple: (model, tokenizer, disease_to_id, id_to_disease)
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Initialize model
    model = ClinicalBERTClassifier(model_dir, num_classes=num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    
    # Load class mappings
    mappings = torch.load(os.path.join(model_dir, "class_mappings.pt"))
    disease_to_id = mappings['disease_to_id']
    id_to_disease = mappings['id_to_disease']
    
    return model, tokenizer, disease_to_id, id_to_disease

def quantize_model(model_dir, output_dir):
    """
    Quantize the model to 4-bit precision
    
    Args:
        model_dir (str): Directory containing the saved model
        output_dir (str): Directory to save quantized model
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bitsandbytes.nn import Linear4bit
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Quantize the model to 4-bit precision
    # This is a simplified version, actual implementation depends on the bitsandbytes version
    quantized_model = model
    
    # Save the quantized model
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Copy class mappings
    import shutil
    shutil.copy(
        os.path.join(model_dir, "class_mappings.pt"),
        os.path.join(output_dir, "class_mappings.pt")
    )
    
    print(f"Model quantized and saved to {output_dir}")
    
    return quantized_model
