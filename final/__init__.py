"""
Singlish vs English Sentiment Analysis Application

This package provides sentiment analysis tools for comparing
Singlish and English text processing with explainability.
"""

__version__ = "1.0.0"

# Import main components
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        pipeline
    )
    import torch
    import shap
    import streamlit as st
    import numpy as np
except ImportError:
    # Dependencies will be installed during deployment
    pass

__all__ = []