from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("text", data_files="character_chatbot/hiccup_lines.txt", split="train")

data = dataset["text"]
train_texts, val_texts = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = Dataset.from_dict({'text': train_texts})
val_dataset = Dataset.from_dict({'text': val_texts})

#tokenizer converts text into tokens that a model can understand
#and reverses the process to decode output
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
tokenizer.pad_token = tokenizer.eos_token

#Tokenize dataset
def tokenize_function(example):
    inputs = tokenizer(example["text"], truncation = True, padding = "max_length", max_length = 128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./hiccup_gpt2",
    logging_dir = "./hiccup_logs",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay = 0.01,
    warmup_steps = 500,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=False,  # set True only if using GPU
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = './model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
