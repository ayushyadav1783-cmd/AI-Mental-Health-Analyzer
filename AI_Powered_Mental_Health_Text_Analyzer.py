from datasets import load_dataset
import pandas as pd

# Load the Hugging Face Emotion dataset
dataset = load_dataset("dair-ai/emotion")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Display some samples
print(train_df.head())
print(train_df["label"].value_counts())

# Optionally save as CSV files
train_df.to_csv("emotion_train.csv", index=False)
val_df.to_csv("emotion_val.csv", index=False)
test_df.to_csv("emotion_test.csv", index=False)

#Next part (for ayush only)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Class distribution
plt.figure(figsize=(8,5))
sns.countplot(x='label', data=train_df, hue='label', palette='viridis', legend=False)
plt.title('Emotion Class Distribution', fontsize=14)
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.show()

# Map numeric labels to names
labels_map = dataset['train'].features['label'].names
train_df['emotion'] = train_df['label'].apply(lambda x: labels_map[x])

# Re-plot with names
plt.figure(figsize=(8,5))
sns.countplot(x='emotion', data=train_df, order=train_df['emotion'].value_counts().index, hue='emotion', palette='coolwarm', legend=False)
plt.title('Emotion Distribution (Named Labels)', fontsize=14)
plt.xticks(rotation=30)
plt.show()

# Text length distribution
train_df['text_length'] = train_df['text'].apply(len)
plt.figure(figsize=(8,5))
sns.histplot(train_df['text_length'], bins=40, kde=True, color='teal')
plt.title('Distribution of Text Lengths', fontsize=14)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Average text length per emotion
plt.figure(figsize=(8,5))
sns.barplot(x='emotion', y='text_length', data=train_df, hue='emotion', palette='magma', legend=False)
plt.title('Average Text Length per Emotion', fontsize=14)
plt.xticks(rotation=30)
plt.show()

print("✅ EDA complete: You can now visualize class balance and text patterns.")

#next part (for ayush only)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1️⃣ Prepare Data
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

# 2️⃣ Tokenizer & Model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# 3️⃣ Training Config
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# 4️⃣ Train
print("🚀 Starting training now...")
trainer.train()
print("✅ Training complete, saving model...")

# 5️⃣ Save the Model
import os
save_dir = "/Users/ayush_home/Downloads/PYTHON/models/emotion_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Model saved successfully at: {save_dir}")

# 6️⃣ Evaluate
preds = trainer.predict(test_dataset)
print(preds.metrics)
print("🎉 All steps finished successfully. Check models/emotion_model directory.")