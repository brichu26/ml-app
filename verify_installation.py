import torch
from transformers import pipeline

print("PyTorch version:", torch.__version__)
summarizer = pipeline("summarization", model="t5-base")
print("Model loaded successfully")

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")