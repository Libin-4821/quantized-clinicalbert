#Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch
import os

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

#Read the Excel file
data = pd.read_excel("symptoms_chest.xlsx", engine="openpyxl")

#number of unique classes
classes = data["Disease"].unique()
num_labels = len(classes)

# Create class to ID mapping
class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
id_to_class = {idx: class_name for idx, class_name in enumerate(classes)}

#Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data["Symptoms"], data["Disease"], test_size=0.3, random_state=9
)

# Prepare datasets
max_input_length = 512

# Tokenize inputs
tokenized_input_train = tokenizer(
    list(X_train),
    max_length=max_input_length,
    truncation=True,
    padding="max_length"
)

tokenized_input_test = tokenizer(
    list(X_test),
    max_length=max_input_length,
    truncation=True,
    padding="max_length"
)

# Convert labels to IDs
y_train_ids = [class_to_id[label] for label in y_train]
y_test_ids = [class_to_id[label] for label in y_test]

# Create datasets
train_dataset = Dataset.from_dict({
    "input_ids": tokenized_input_train["input_ids"],
    "attention_mask": tokenized_input_train["attention_mask"],
    "labels": y_train_ids
})

validation_dataset = Dataset.from_dict({
    "input_ids": tokenized_input_test["input_ids"],
    "attention_mask": tokenized_input_test["attention_mask"],
    "labels": y_test_ids
})

#Loading the pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    "medicalai/ClinicalBERT",
    num_labels=num_labels
)

#Training arguments
training_args = TrainingArguments(
    output_dir="./results_full",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=500,
)

#Loading the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained full precision model
model.save_pretrained("clinicalbert-trained")
tokenizer.save_pretrained("clinicalbert-trained")

