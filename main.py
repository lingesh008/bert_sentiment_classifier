from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

# Step 1: Load IMDB dataset
print("📥 Loading IMDB dataset...")
dataset = load_dataset("imdb")

# ✅ Optional: Reduce test size for faster evaluation
dataset["test"] = dataset["test"].select(range(500))  # Evaluate on 500 samples only

# Step 2: Load BERT tokenizer
print("🔡 Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Step 3: Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Step 4: Tokenize the dataset
print("🧠 Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 5: Set format for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Step 6: Create dataloaders
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=16)  # Increased batch size for speed

# Step 7: Show sample batch
print("\n✅ Sample batch from train dataloader:")
for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break

# Step 8: Load BERT for classification
print("\n📦 Loading BERT model for sequence classification...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 9: Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 10: Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 11: Training (1 batch demo)
print("\n🚀 Starting training...")
model.train()

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["label"]
    )

    loss = outputs.loss
    logits = outputs.logits

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"🔁 Training loss: {loss.item():.4f}")
    break  # Remove break to train full dataset

# Step 12: Evaluation
print("\n🔍 Evaluating on test set...")
model.eval()

predictions = []
true_labels = []

with torch.inference_mode():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["label"].cpu().numpy())

# Step 13: Accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"\n✅ Test Accuracy (on 500 samples): {accuracy:.4f}")
