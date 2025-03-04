
# Quantized ClinicalBERT for Disease Classification(Chest related diseases)

This repository contains a 4-bit quantized ClinicalBERT model for disease classification based on clinical text. Inspired by CheXNet, this model can predict diseases from patient symptom descriptions, particularly focusing on chest-related conditions.


## Overview
The model uses a quantized version of ClinicalBERT to classify disease conditions based on clinical text descriptions. Quantization reduces the model size and memory requirements while maintaining performance, making it more efficient for deployment.

## Dataset

This project uses a synthetic dataset of clinical text descriptions for training. The synthetic data mimics real-world clinical notes while avoiding privacy concerns associated with actual patient data. The dataset includes various chest-related conditions with their corresponding symptom descriptions.
## Features
* 4-bit quantized ClinicalBERT for efficient inference.

* Disease classification from symptom descriptions.

* Flask (Python) web interface for easy interaction.

* Compatible with Google Colab for accessible deployment.

## Installation 

```bash
# Clone the repository
git clone https://github.com/john-osborne-j  quantized-clinicalbert.git
 
cd quantized-clinicalbert
```
    
## Requirements
```bash
# Install dependencies

pip install -r requirements.txt
```
## Model Training

The model was trained by first fine-tuning ClinicalBERT on a dataset of chest-related disease descriptions, then quantizing the trained model to 4-bit precision using the BitsAndBytes library.
## Quantization Process

1.Train the full-precision model on the disease classification dataset.

2.Quantize the trained model to 4-bit precision for efficient inference.

3.Save both the class mapping and the quantized model.


## Authors

* [@john-osborne-j](https://github.com/john-osborne-j) 😁
* [@Libin-4821](https://github.com/Libin-4821) 😎


## Usage/Examples

```javascript
symptoms_text = "Progressive shortness of breath over several months, now worse with minimal exertion. Chronic productive cough especially in the mornings with clear to white sputum. Reports chest tightness but no sharp pain. Long history of smoking 1 pack per day for 30 years."

predicted_disease = predict_disease(symptoms_text, model_path="clinicalbert-4bit-quantized")
print(f"Predicted disease: {predicted_disease}")
```

```javascript
output:
`low_cpu_mem_usage` was None, now default to True since model is quantized.
Predicted disease: COPD

```
## Demo

![Image](https://github.com/user-attachments/assets/57c00e81-e26f-4c2e-bbe1-8adae9d74bd7)

![Image](https://github.com/user-attachments/assets/ccf19d57-a0d4-4fe5-88a7-d30920b0416f)

![Image](https://github.com/user-attachments/assets/bf11a309-ab09-4229-ac8e-ed2213fa52ca)

## Limitations

* This model should not replace professional medical diagnosis.
* Performance depends on how closely the  input matches the format and vocabulary of the training data.
* The 4-bit quantization may result in slight accuracy degradation compared to the full-precision model.
## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

* This project was inspired by CheXNet for chest X-ray diagnosis.
* Uses the ClinicalBERT model from (https://github.com/huggingface/transformers) .
* Quantization implemented using the (https://github.com/TimDettmers/bitsandbytes) library.



