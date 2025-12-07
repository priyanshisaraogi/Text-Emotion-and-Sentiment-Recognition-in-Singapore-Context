"""
Singlish vs English Sentiment Analysis Application

This package provides sentiment analysis tools for comparing
Singlish and English text processing.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components if needed
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    import shap
    import streamlit as st
    import numpy as np
except ImportError:
    pass  # Dependencies will be installed during deployment