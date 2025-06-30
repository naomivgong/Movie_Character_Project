from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import json

MODEL_ID = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    return model, tokenizer

def batch_formatter(batch):
    texts = [f'Q: {q}\n\nA: {a}' for q, a in zip(batch["prompt"], batch["response"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

if __name__ == "__main__":
    path = "character_chatbot/hiccup_qa.json"
    model, tokenizer = tokenize_and_model()
    
    dataset = Dataset.from_json(path)
    tokenized_dataset = dataset.map(batch_formatter, batched=True, num_proc = 4)
