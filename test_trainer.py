from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

print("Torch version:", torch.__version__)

# tiny sample dataset
dataset = load_dataset("dair-ai/emotion", split="train[:100]")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=5
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
pass
print("✅ TRAINER WORKS")