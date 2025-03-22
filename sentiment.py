import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# Load dataset and sample 6000 rows
file_path = "C:/Users/rv401/Downloads/IMDB Dataset.csv/IMDB Dataset.csv"  # Update with your file path
dataset = pd.read_csv(file_path)
dataset = dataset.sample(6000, random_state=42)  # Sample 6000 rows for training/testing

# Print dataset sample
print(dataset.head())

# Encode sentiment labels
label_encoder = LabelEncoder()
dataset['sentiment'] = label_encoder.fit_transform(dataset['sentiment'])

# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['review'], dataset['sentiment'], test_size=0.2, random_state=42
)

# Tokenize reviews using RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=512, return_tensors="pt"
)
test_encodings = tokenizer(
    list(test_texts), truncation=True, padding=True, max_length=512, return_tensors="pt"
)

# Convert labels to tensors
train_labels = torch.tensor(train_labels.values)
test_labels = torch.tensor(test_labels.values)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(
    train_encodings['input_ids'], train_encodings['attention_mask'], train_labels
)
test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'], test_labels
)

batch_size = 8  # Optimal for CPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set device to CPU
device = torch.device('cpu')
model.to(device)

# Training
epochs = 3  # Adjusted for CPU training
model.train()

for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    total_loss = 0

    for batch in loop:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy:.2f}')

# Sentiment Prediction Function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(
        text, truncation=True, padding='max_length', max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_encoder.inverse_transform([prediction])[0]

# Interactive Sentiment Prediction
while True:
    user_input = input("Enter a review (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(f"Predicted Sentiment: {predict_sentiment(user_input)}")
