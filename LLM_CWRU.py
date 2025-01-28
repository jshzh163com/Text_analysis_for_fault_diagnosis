# -*- coding: utf-8 -*-


from models.CNN_1 import CNN
from sklearn.neighbors import KNeighborsClassifier as KNN
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Tokenizer
tokenizer = get_tokenizer("basic_english")


def yield_tokens(sentences):
    for sentence in sentences:
        yield tokenizer(sentence)


input_file = "CWRU_sentences_with_labels_load_1.txt"

sentences = []
labels = []

# Read the file line by line
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if "Label: " in line:
            sentence, label = line.rsplit("Label: ", 1)
            sentences.append(sentence.strip())
            labels.append(int(label))

vocab = build_vocab_from_iterator(
    yield_tokens(sentences), specials=["<unk>", "<pad>"])

indexed_sentences = [vocab(tokenizer(sentence)) for sentence in sentences]
ind_sentences = np.array(indexed_sentences)


train_data, test_data, train_label, test_label = train_test_split(
    ind_sentences, labels, test_size=0.2, shuffle=True)

knn = KNN(n_neighbors=10, algorithm='auto',
          weights='distance', leaf_size=30,
          metric='minkowski', p=2,
          metric_params=None, n_jobs=1)

train_knn = knn.fit(train_data, train_label)
train_knn.score(train_data, train_label)
train_knn.score(test_data, test_label)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # Convert to PyTorch tensor
        self.data = torch.tensor(data, dtype=torch.float32)
        # Convert to PyTorch tensor
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = CustomDataset(ind_sentences, labels)

train_dataset, test_dataset = train_test_split(
    dataset, test_size=0.2, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 5 layers
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

for epoch in range(50):
    model.train()
    total_loss = 0.0
    epoch_acc = 0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        correct = torch.eq(output.argmax(dim=1), labels).float().sum().item()
        epoch_acc += correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader.dataset)
    epoch_acc /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {epoch_acc:.4f}")

model.eval()
epoch_acc_tr = 0
correct_train = 0
with torch.no_grad():
    total = 0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        output = model(inputs)
        correct_train = torch.eq(output.argmax(
            dim=1), labels).float().sum().item()
        epoch_acc_tr += correct_train
    epoch_acc_tr /= len(train_loader.dataset)
    print(f"Train acc: {epoch_acc_tr:.4f}")

correct = 0
with torch.no_grad():
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        output = model(inputs)
        correct = torch.eq(output.argmax(dim=1), labels).float().sum().item()
        epoch_acc += correct
    epoch_acc /= len(test_loader.dataset)
    print(f"Test acc: {epoch_acc:.4f}")
