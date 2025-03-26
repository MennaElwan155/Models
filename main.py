import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(0)

# Load Dataset
df = pd.read_csv(r'C:\\Users\Computec\Desktop\diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

df = df.astype(int)

# Convert DataFrame to Torch Tensors
data = torch.tensor(df.values, dtype=torch.float32)
X, y = data[:, 1:].to(device), data[:, 0].long().to(device)

# Shuffle indices
torch.manual_seed(0)
indices = torch.randperm(X.size(0))

# Split dataset
train_val_split = int(0.9 * X.size(0))
val_split = int(0.2 * train_val_split)
train_idx, val_idx, test_idx = indices[:train_val_split], indices[train_val_split:], indices[:val_split]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize manually
min_vals, max_vals = X_train.min(dim=0, keepdim=True)[0], X_train.max(dim=0, keepdim=True)[0]
X_train = (X_train - min_vals) / (max_vals - min_vals)
X_val = (X_val - min_vals) / (max_vals - min_vals)
X_test = (X_test - min_vals) / (max_vals - min_vals)

# Define Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(21, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        return self.layers(x)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

def train_model(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs=500):
    train_loss, val_loss = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=1) == y_train).float().mean().item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss_val = loss_fn(val_logits, y_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            val_loss.append(val_loss_val.item())
        
        train_loss.append(loss.item())
        if epoch % 50 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%, Val Loss: {val_loss_val.item():.4f}, Val Acc: {val_acc*100:.2f}%')
    return train_loss, val_loss

train_loss, val_loss = train_model(model, optimizer, loss_fn, X_train, y_train, X_val, y_val)

plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on Test Set
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_loss = loss_fn(test_logits, y_test).item()
    test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
    
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc*100:.2f}%')
