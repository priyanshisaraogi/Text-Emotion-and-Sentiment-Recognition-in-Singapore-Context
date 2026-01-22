# NLP Final Project: Singlish Sentiment Analysis

## Project Overview

This project explores sentiment analysis on Singlish (Singaporean English), a colloquial form of English incorporating Malay, Chinese, and other influences. We demonstrate the importance of analyzing Singlish directly rather than translating it to standard English, as translation often loses the nuanced meanings conveyed by slang and cultural references.

The project compares two approaches:
1. **Translation Path**: Singlish sentence → English translation → English sentiment analysis
2. **Direct Path**: Singlish sentence → Direct Singlish sentiment analysis

This comparison highlights how translation can diminish the emotional and contextual depth of Singlish. Additionally, we use SHAP (SHapley Additive exPlanations) to analyze token importance in sentiment predictions.

## Codebase Structure

- **`datasets/`**: Contains all datasets used for training the models.
  - `final_training_translator(HF Dataset+gpt-generated).csv`: Dataset for translation model training.
  - `GoEmotions_dataset_sample.csv`: Sample from GoEmotions dataset.
  - `singlish_sentiment.csv`: Singlish sentiment dataset.

- **`final/`**: Contains the production-ready code.
  - `app.py`: Streamlit application for running the models from Hugging Face and displaying results on a web interface.
  - `requirements.txt`: Lists all Python dependencies required to run the application.


- **`__init__.py`**: Python package initialization file.

The trained models are uploaded to Hugging Face. Trial code and final model training scripts were executed in Google Colab. PDFs of the project report and a demo video are included in the repository.

## Installation and Running Instructions

### Prerequisites
- Python 3.12
- pip (Python package installer)

### Steps

1. **Navigate to the final folder**:
   ```
   cd final
   ```

2. **Create a virtual environment**:
   - **Mac/Linux**:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```
   python -m streamlit run app.py
   ```

The application will launch in your default web browser, allowing you to interact with the sentiment analysis models.
