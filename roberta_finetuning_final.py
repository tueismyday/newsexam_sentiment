from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split

#######################################
# File Paths
#######################################

labeled_file_path = "data/excluded_wallstreetbets_data_labels.csv"  # Labeled excluded rows
output_dir = "./roberta-finetuned-sentiment"  # Output directory for the fine-tuned model
os.makedirs(output_dir, exist_ok=True)

#######################################
# System Resource Check
#######################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#######################################
# Load Labeled Excluded Data
#######################################

print("\nLoading labeled excluded data...")
labeled_df = pd.read_csv(labeled_file_path)

# Verify the dataset
if 'label' not in labeled_df.columns:
    raise ValueError("The dataset must have a 'label' column with 'positive', 'neutral', or 'negative' values.")

# Combine 'title' and 'body' for input text
labeled_df['text'] = labeled_df[['title', 'body']].fillna("").agg(" ".join, axis=1)
labeled_df = labeled_df[['text', 'label']]  # Keep only relevant columns

# Drop rows with NaN labels
labeled_df = labeled_df.dropna(subset=['label'])
print(f"Loaded {len(labeled_df)} rows of labeled data after dropping NaNs.")

#######################################
# Count Distribution of Labels
#######################################

print("\nLabel distribution:")
label_counts = labeled_df['label'].value_counts()
print(label_counts)

#######################################
# Encode Labels
#######################################

label_mapping = {"positive": 2, "neutral": 1, "negative": 0}
labeled_df['label'] = labeled_df['label'].map(label_mapping)

#######################################
# Split Data into Training and Evaluation Sets
#######################################

train_df, eval_df = train_test_split(
    labeled_df,
    test_size=0.2,
    random_state=42,
    stratify=labeled_df['label']  # Stratify based on the label
)
print(f"\nTraining data size: {len(train_df)}")
print(f"Evaluation data size: {len(eval_df)}")

#######################################
# Tokenization
#######################################

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """
    Tokenizes the text data and includes labels.
    """
    return tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("\nTokenizing data...")
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

#######################################
# Load Model for Sequence Classification
#######################################

num_labels = 3  # Positive, Neutral, Negative
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

#######################################
# Training Arguments
#######################################

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=20,  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=1,  # No need for gradient accumulation
    save_steps=50,  
    save_total_limit=1,  # Keep only the last checkpoint
    logging_dir="./logs",
    logging_steps=50,  
    learning_rate=1e-5,  # Low learning rate for fine-tuning small data
    weight_decay=0.01,
    warmup_steps=10,  # Few warmup steps
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    save_strategy="epoch"  # Save only at the end of epochs
)

#######################################
# Initialize Trainer
#######################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # Add evaluation dataset
    tokenizer=tokenizer,
)

#######################################
# Fine-Tuning
#######################################

print("\nStarting fine-tuning on labeled excluded rows...")
trainer.train()

#######################################
# Save the Final Model
#######################################

print("\nSaving the fine-tuned sentiment analysis model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}.")

print("\nFine-tuning completed successfully!")
