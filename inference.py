#Importing necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch
import os
import json

def predict_disease(symptoms_text, model_path="clinicalbert-4bit-quantized"):
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load class mapping
    with open(f"{model_path}/class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    id_to_class = class_mapping["id_to_class"]

    # Convert string keys back to integers (JSON serializes all keys as strings)
    id_to_class = {int(k): v for k, v in id_to_class.items()}

    # Configure quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        #device_map="auto"
    )

    # Prepare input
    inputs = tokenizer(
        symptoms_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(model.device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_disease = id_to_class[predicted_class_id]

    return predicted_disease
