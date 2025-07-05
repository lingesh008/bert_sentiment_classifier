🎯 BERT Sentiment Classifier

This project is a sentiment analysis model built using **BERT** and the **IMDB movie reviews dataset**, powered by **HuggingFace Transformers** and **PyTorch**.

📌 Features

- ✅ Uses `bert-base-uncased` for fine-tuning
- ✅ Classifies text as **Positive** or **Negative**
- ✅ IMDB dataset integration using 🤗 `datasets`
- ✅ Lightweight training demo (1 batch)
- ✅ Accuracy evaluation with `scikit-learn`
- ✅ Includes tokenization, training, and testing pipelines

---

🛠️ Requirements

Install dependencies:
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

📊 Example Output

📥 Loading IMDB dataset...
🔁 Training loss: 0.6311
🔍 Evaluating on test set...
✅ Test Accuracy (on 500 samples): 0.8854
