import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ========== Set Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Data Paths ==========
train_dir = "Group Ass/Aerial-Landscape-master/Aerial_Landscapes_split/train"
test_dir  = "Group Ass/Aerial-Landscape-master/Aerial_Landscapes_split/test"

# ========== Image Preprocessing ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== Load Data ==========
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = len(train_dataset.classes)

# ========== Load Pre-trained ResNet50 and Extract Features ==========
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            output = resnet(imgs) 
            output = output.view(output.size(0), -1) 
            features.append(output.cpu())
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

print("Extracting training features...")
train_features, train_labels = extract_features(train_loader)
print("Extracting testing features...")
test_features, test_labels = extract_features(test_loader)

print("Training feature shape:", train_features.shape)
print("Testing feature shape:", test_features.shape)

# ========== Define Attention Module ==========
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn = self.fc1(x)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        return x * attn

# ========== Define Improved MLP Classifier ==========
class MLPClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifierWithAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.attn = AttentionBlock(hidden_dim, reduction=8)  # reduction parameter can be adjusted
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):  # attention module is applied here
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.attn(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

input_dim = train_features.shape[1]  # 2048
hidden_dim = 512                     # can modify as needed
model = MLPClassifierWithAttention(input_dim, hidden_dim, num_classes).to(device)

# ========== Define Loss Function and Optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_features = train_features.to(device)
train_labels = train_labels.to(device)
test_features = test_features.to(device)
test_labels = test_labels.to(device)
 
# ========== Train MLP ==========
num_epochs = 20
batch_size = 100
num_train = train_features.size(0)
loss_history = []
train_acc_history = []
test_acc_history = []

model.train()
for epoch in range(num_epochs):
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    for i in range(0, num_train, batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = train_features[indices]
        batch_y = train_labels[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    loss_history.append(epoch_loss)
    model.eval()
    with torch.no_grad():
        outputs_train = model(train_features)
        _, preds_train = torch.max(outputs_train, dim=1)
        train_acc = accuracy_score(train_labels.cpu(), preds_train.cpu())
        
        outputs_test = model(test_features)
        _, preds_test = torch.max(outputs_test, dim=1)
        test_acc = accuracy_score(test_labels.cpu(), preds_test.cpu())
    
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.3f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    model.train()

# ========== Plot Training Loss Curve ==========
plt.figure()
plt.plot(loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# ========== Plot Accuracy Curve (Train & Test) ==========
plt.figure()
epochs = range(1, num_epochs+1)
plt.plot(epochs, train_acc_history, marker='o', label='Train Accuracy')
plt.plot(epochs, test_acc_history, marker='o', label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# ========== Model Evaluation ==========
model.eval()
with torch.no_grad():
    outputs = model(test_features)
    _, preds = torch.max(outputs, 1)
    
acc = accuracy_score(test_labels.cpu(), preds.cpu())
print(f"Final Test Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(test_labels.cpu(), preds.cpu())
print("Confusion Matrix:")
print(cm)

report = classification_report(test_labels.cpu(), preds.cpu(), target_names=train_dataset.classes)
print("Classification Report:")
print(report)

# ========== Plot Confusion Matrix ==========
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

plot_confusion_matrix(cm, classes=train_dataset.classes, normalize=False, title='Confusion Matrix', save_path="confusion_matrix.png")
