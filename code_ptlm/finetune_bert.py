import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from datasets import load_dataset

train_data = load_dataset("bigcode/stack", data_dir="data/python", split="train", streaming=True)
test_data = load_dataset("bigcode/stack", data_dir="data/python", split="test", streaming=True)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_enc = tokenizer()


