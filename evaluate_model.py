from transformers import Trainer
import torch

# Load the trained model and validation dataset
model = torch.load("./results/checkpoint-final")  # Path to the last checkpoint or final model
trainer = Trainer(model=model)

# Load validation dataset
val_dataset = torch.load("val_dataset.pt")

# Evaluate the model
eval_results = trainer.evaluate(val_dataset)
print(f"Evaluation results: {eval_results}")
