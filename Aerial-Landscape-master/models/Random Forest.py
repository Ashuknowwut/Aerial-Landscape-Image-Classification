import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== Setup ==========
train_dir = "Group Ass/Aerial-Landscape-master/Aerial_Landscapes_split/train"
test_dir = "Group Ass/Aerial-Landscape-master/Aerial_Landscapes_split/test"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Image Preprocessing ==========
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Load Data ==========
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

label_map = train_dataset.classes

# ========== Load Pre-trained ResNet50 and Extract Features ==========
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

# ========== function of Feature Extraction ==========
def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            output = resnet(imgs).view(imgs.size(0), -1).cpu().numpy()
            features.extend(output)
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

# ========== Feature Extraction ==========
print("Extracting training features...")
X_train, y_train = extract_features(train_loader)
print("Extracting testing features...")
X_test, y_test = extract_features(test_loader)
print("Feature dimension:", X_train.shape[1], "Number of training samples:", len(X_train))

# ========== Dimensionality Reduction ==========
pca = PCA(n_components=256)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("Reduced feature dimension:", X_train_pca.shape[1])

# ========== Train Random Forest ==========

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8, 
    min_samples_split=4,
    min_samples_leaf=2,      
    random_state=42
)
clf.fit(X_train_pca, y_train)

# ========== Model Evaluation ==========
y_pred = clf.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}")

report = classification_report(y_test, y_pred, target_names=label_map)
print("\nClassification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

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

plot_confusion_matrix(cm, classes=label_map, normalize=False, title='Confusion Matrix', save_path="confusion_matrix.png")

# Accuracy Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train_pca, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Accuracy")
plt.plot(train_sizes, val_scores_mean, 'o-', label="Validation Accuracy")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.title("Accuracy Learning Curve")
plt.legend(loc="best")
plt.savefig("learning_curve_accuracy.png", dpi=300)
plt.show()

# Log Loss Learning Curve
train_sizes_loss, train_scores_loss, val_scores_loss = learning_curve(
    clf, X_train_pca, y_train, cv=5, scoring='neg_log_loss', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_loss = -np.mean(train_scores_loss, axis=1)
val_loss = -np.mean(val_scores_loss, axis=1)

plt.figure()
plt.plot(train_sizes_loss, train_loss, 'o-', label="Training Loss")
plt.plot(train_sizes_loss, val_loss, 'o-', label="Validation Loss") 
plt.xlabel("Number of training samples")
plt.ylabel("Log Loss")
plt.title("Log Loss Learning Curve")
plt.legend(loc="best")
plt.savefig("learning_curve_loss.png", dpi=300)
plt.show()
