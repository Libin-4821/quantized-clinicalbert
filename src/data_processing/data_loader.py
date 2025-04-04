import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

class DiseaseSymptomDataset(Dataset):
    """Dataset for disease symptoms classification"""
    
    def __init__(self, data_file, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", max_length=512):
        """
        Initialize the dataset
        
        Args:
            data_file (str): Path to CSV file with symptoms and disease labels
            tokenizer_name (str): Name of the HuggingFace tokenizer to use
            max_length (int): Maximum sequence length for tokenizer
        """
        self.data = pd.read_csv(data_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Create label mapping
        self.diseases = self.data['disease'].unique()
        self.disease_to_id = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.id_to_disease = {idx: disease for disease, idx in self.disease_to_id.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        symptoms = self.data.iloc[idx]['symptoms']
        disease = self.data.iloc[idx]['disease']
        
        # Tokenize the symptoms
        encoding = self.tokenizer(
            symptoms,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Get the label
        label = self.disease_to_id[disease]
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(data_file, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", 
                    batch_size=16, train_split=0.8, val_split=0.1, 
                    random_state=42, max_length=512):
    """
    Create train, validation and test dataloaders
    
    Args:
        data_file (str): Path to CSV file with symptoms and disease labels
        tokenizer_name (str): Name of the HuggingFace tokenizer to use
        batch_size (int): Batch size for dataloaders
        train_split (float): Proportion of data to use for training
        val_split (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
        max_length (int): Maximum sequence length for tokenizer
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data
    data = pd.read_csv(data_file)
    
    # Split data
    train_data = data.sample(frac=train_split, random_state=random_state)
    temp_data = data.drop(train_data.index)
    
    val_size = val_split / (1 - train_split)
    val_data = temp_data.sample(frac=val_size, random_state=random_state)
    test_data = temp_data.drop(val_data.index)
    
    # Save split datasets
    train_data.to_csv('data/processed/train_data.csv', index=False)
    val_data.to_csv('data/processed/val_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    # Create datasets
    train_dataset = DiseaseSymptomDataset('data/processed/train_data.csv', tokenizer_name, max_length)
    val_dataset = DiseaseSymptomDataset('data/processed/val_data.csv', tokenizer_name, max_length)
    test_dataset = DiseaseSymptomDataset('data/processed/test_data.csv', tokenizer_name, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, train_dataset.disease_to_id, train_dataset.id_to_disease

def convert_symptoms_to_df(symptoms_data):
    """
    Convert the symptoms data to a pandas DataFrame
    
    Args:
        symptoms_data (str): String containing symptoms data in the format 
                           "ID Disease Symptoms"
                           
    Returns:
        pandas.DataFrame: DataFrame with columns 'id', 'disease', 'symptoms'
    """
    rows = []
    for line in symptoms_data.strip().split('\n'):
        parts = line.split(' ', 2)
        if len(parts) == 3:
            id_num, disease, symptoms = parts
            rows.append({
                'id': id_num,
                'disease': disease,
                'symptoms': symptoms
            })
    
    return pd.DataFrame(rows)

def save_sample_data():
    """
    Create and save a sample dataset based on the symptom examples
    """
    sample_data = """
    4566 Heart_Failure Swollen legs, increasing swollen ankles, increasing shortness of breath, fatigue, rapid heartbeat, difficulty breathing when lying down, ascites.
    1897 Pneumonia Sweating connected with chest pain (markedly moderate) in combination with confusion related to shortness of breath (generally intense).
    7301 Lung_Cancer Persistent cough and weight loss accompanied by neurological symptoms and weakness connected with chest pain and deep vein thrombosis in conjunction with unexplained fever and facial swelling.
    5663 COPD Severe persistent cough, wheezing, pursed-lip breathing with mild prolonged expiration, worse in the morning.
    3323 Tuberculosis Coughing up blood, night sweats, fever, mild loss of appetite, unintended weight loss, worse when lying down.
    3299 Tuberculosis Persistent cough, low-grade night sweats, weight loss, low-grade coughing up blood, blood in mucus, unintended weight loss, hemoptysis, severe pleural effusion, worse in the morning.
    2939 Tuberculosis Feverish at night, coughing up blood and joint pain correlated with fatigue.
    """
    
    df = convert_symptoms_to_df(sample_data)
    
    # Make sure the directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save the dataframe
    df.to_csv('data/raw/symptoms_data.csv', index=False)
    
    return df

if __name__ == "__main__":
    # Create sample data
    sample_df = save_sample_data()
    print(f"Sample data created with {len(sample_df)} entries.")
    print(sample_df.head())
