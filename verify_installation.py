import torch
from transformers import pipeline

print("PyTorch version:", torch.__version__)
summarizer = pipeline("summarization", model="t5-base")
print("Model loaded successfully")