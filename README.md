# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="826" height="765" alt="image" src="https://github.com/user-attachments/assets/79bc15f6-3308-4ae2-8b81-ca9bd8f4f8c8" />

## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\admin\Downloads\customers.csv")

# Drop ID column
df.drop(columns=["ID"], inplace=True)


df["Ever_Married"].fillna("No", inplace=True)
df["Graduated"].fillna("No", inplace=True)
df["Profession"].fillna("Unknown", inplace=True)
df["Var_1"].fillna("Unknown", inplace=True)
df["Work_Experience"].fillna(0, inplace=True)
df["Family_Size"].fillna(df["Family_Size"].median(), inplace=True)

print("Missing Values")
print(df.isnull().sum())


cat_cols = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1",
    "Segmentation"
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


X = df.drop(columns=["Segmentation"])
y = df["Segmentation"]


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X_train = torch.tensor(X_train.copy(), dtype=torch.float32)
X_test = torch.tensor(X_test.copy(), dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(copy=True), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(copy=True), dtype=torch.long)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Classifier(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


model = Classifier(X_train.shape[1])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 100

print("\nTraining Started...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss = {running_loss/len(train_loader):.4f}")


model.eval()

actual = []
predicted = []

with torch.no_grad():

    for features, labels in test_loader:

        outputs = model(features)

        _, preds = torch.max(outputs, 1)

        predicted.extend(preds.cpu().numpy())

        actual.extend(labels.cpu().numpy())


accuracy = accuracy_score(actual, predicted)

print("\nAccuracy :", accuracy)


cm = confusion_matrix(actual, predicted)

print("\nConfusion Matrix\n")
print(cm)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A", "B", "C", "D"],
    yticklabels=["A", "B", "C", "D"]
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Customer Segmentation Confusion Matrix")

plt.show()


print("\nClassification Report\n")

print(classification_report(actual, predicted))
```

### OUTPUT
<img width="1036" height="210" alt="image" src="https://github.com/user-attachments/assets/040eaad4-6031-43aa-86e9-a35b8007cdca" />
<img width="547" height="51" alt="image" src="https://github.com/user-attachments/assets/e7d78cae-6583-4094-805c-166f936bc1f5" />

## Confusion Matrix
<img width="557" height="77" alt="image" src="https://github.com/user-attachments/assets/33e0c985-15bb-4087-be3d-fc65a4437304" />

## Classification Report
<img width="822" height="546" alt="image" src="https://github.com/user-attachments/assets/30728085-2eef-4475-a7e9-0eacd17a084f" />

### New Sample Data Prediction
<img width="568" height="87" alt="image" src="https://github.com/user-attachments/assets/952befe8-a2d9-4993-aa04-46a7d14799d2" />

## RESULT
This program has been executed successfully.


Include your result here
