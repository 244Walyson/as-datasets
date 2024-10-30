from transformers import AutoTokenizer
import torch

# Load the trained model and tokenizer
model = torch.load("./results/checkpoint-final")  # Path to the last checkpoint or final model
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Save the model and tokenizer
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
print("Model and tokenizer saved in './sentiment_model'")
