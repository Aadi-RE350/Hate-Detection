import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. CONFIG
MODEL_CKPT = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
# MAX_LEN * BATCH_SIZE = 128 * 32 = 4096  

EPOCHS = 3
LEARNING_RATE = 1e-3 # Higher LR because we are only training the CNN part


# 2. DATA LOADING 
print("Loading Tokenizer and Data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

# Load CSVs
train_df = pd.read_csv("csv/train.csv")
test_df = pd.read_csv("csv/test.csv")

class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['cleaned_text'].astype(str).values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Create DataLoaders
train_dataset = BertDataset(train_df, tokenizer, MAX_LEN)
test_dataset = BertDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3. MODEL (CNN + BERT)
class BertCNN(nn.Module):
    def __init__(self, freeze_bert=True):
        super(BertCNN, self).__init__()
        
        # LOAD BERT
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        
        # FREEZE BERT 
        if freeze_bert:
            print("Freezing BERT parameters...")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # CNN LAYERS
        # BERT outputs 768 dimensions per word.
        # We treat these 768 dims as "Channels" for the CNN.
        self.conv = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        
        # CLASSIFIER
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1) # Output 1 score
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        

        x = last_hidden_state.permute(0, 2, 1)
        
        
        x = self.conv(x)      # Convolve
        x = self.relu(x)      # Activate
        x = self.pool(x)      # Pool (get max features)
        
        
        x = x.squeeze(2)      # Remove extra dimension
        x = self.dropout(x)
        x = self.fc(x)
        
        return self.sigmoid(x)

# Initialize
model = BertCNN(freeze_bert=True).to(device)

# 4. TRAINING LOOP

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print("\nStarting Training (BERT Frozen)...")

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    n_batches = len(data_loader)
    
    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        # Forward
        outputs = model(input_ids, attention_mask)
        
        preds = (outputs > 0.5).float().squeeze(1)
        
        loss = loss_fn(outputs.squeeze(1), targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        # Print batch progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f'  Batch {i + 1}/{n_batches} | Loss: {loss.item():.4f}')
        
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            preds = (outputs > 0.5).float().squeeze(1)
            
            loss = loss_fn(outputs.squeeze(1), targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# --- Run Epochs ---
for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc, val_loss = eval_model(model, test_loader, criterion, device)
    
    print(f'Epoch {epoch + 1}/{EPOCHS} | Time: {time.time() - start_time:.1f}s')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%')
    print("-" * 30)

# Save Model
torch.save(model.state_dict(), 'bert_cnn_frozen.pth')
print("Model saved to bert_cnn_frozen.pth")