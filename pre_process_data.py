import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# Load dataset
df = pd.read_csv('dataset.csv', usecols=['text', 'polarity'])  # Load only necessary columns

# Check for non-finite values in 'polarity' and handle them
df = df[df['polarity'].notna()]  # Drop rows where 'polarity' is NaN
df['polarity'] = df['polarity'].replace([float('inf'), float('-inf')], 0)  # Replace inf/-inf with 0 (or another value)

# Ensure the 'text' column does not contain NaN values
df = df[df['text'].notna()]  # Drop rows where 'text' is NaN

texts = df['text'].tolist()
labels = df['polarity'].astype(int).tolist()  # Convert labels to integers after cleaning

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load BERTimbau tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Tokenize in batches to reduce memory usage
def tokenize_in_batches(texts, batch_size=32):
    encodings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Ensure batch is a list of valid strings
        batch = [text for text in batch if isinstance(text, str)]  # Filter out non-string values
        if len(batch) > 0:  # Only tokenize if batch is not empty
            encodings.extend(tokenizer(batch, truncation=True, padding=True, max_length=128)['input_ids'])
        else:
            print(f"Empty batch found at index {i}")
    return encodings

train_encodings = tokenize_in_batches(train_texts)
val_encodings = tokenize_in_batches(val_texts)

# Save processed data to avoid recomputation
torch.save((train_encodings, train_labels, val_encodings, val_labels), "processed_data.pt")
print("Data preprocessed and saved to 'processed_data.pt'")

# Clear variables to free memory
del train_texts, val_texts, train_labels, val_labels, train_encodings, val_encodings
torch.cuda.empty_cache()  # If you're using a GPU, clear the cache
