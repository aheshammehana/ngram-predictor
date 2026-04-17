# N-Gram Next-Word Predictor (capstone provided by A.Kashef)

**Author:** Ahmed Mehana

## Project Overview
This project implements a **next-word prediction model** based on an **n-gram language model**.  
The model is trained on four Sherlock Holmes novels from Project Gutenberg and predicts the most likely next word given a text prompt.

The system is built **from scratch**, without external NLP libraries, and follows a clean modular pipeline:
- Data preparation
- Model training
- Inference with backoff
- Command-line interface (CLI)

---

## Project Structure

ngram-predictor/
├── main.py
├── config/
│   └── .env
├── src/
│   ├── data_prep/
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   └── inference/
│       └── predictor.py
├── data/            # generated locally (ignored by Git)
└── README.md

## Setup Instructions

1. Create and activate a Python environment (Anaconda is recommended):

conda create -n ngram python=3.10
conda activate ngram

2. Install required dependencies:
    pip install -r requirements.txt
    
4. Ensure the configuration file exists:
   config/.env

5. Running the Project (CLI):
   1. Step 1 — Data Preparation
      python main.py --step dataprep
      output:
      data/processed/train_tokens.txt  
   2.  Model Training: Builds the vocabulary, n-gram counts, and computes MLE probabilities
      python main.py --step model
      Output:
       data/model/vocab.json
       data/model/model.json
   3.  Inference (Prediction): Run the interactive next-word predictor.
        python main.py --step inference
        Output:
        > sherlock holmes
        Predictions: said, was, had
        > quit

6. Model Details
    N-gram order: Configurable (default: 4)
    Training method: Maximum Likelihood Estimation (MLE)
    Smoothing: None (backoff only)
    OOV handling: <UNK> token
    Inference: Backoff from highest to lowest n-gram order
7. Notes
    Generated data and models are excluded from Git via .gitignore.
    The model must be trained before running inference.
    The project is designed to be run entirely from the command line.

8. License
   Educational use



