"""Reference run: same task, architecture and hyperparameters as neural_network.cpp.

Usage:  python compare_pytorch.py [seed]
Needs:  pip install torch numpy
"""
import sys
import time

import numpy as np
import torch
from torch import nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

# Same data distribution as the C++ demo: 2000 points in [-2, 2]^2, label = inside unit circle.
X = rng.uniform(-2.0, 2.0, size=(2000, 2))
y = (np.linalg.norm(X, axis=1) < 1.0).astype(np.float32).reshape(-1, 1)

perm = rng.permutation(len(X))
n_test = int(0.2 * len(X))
test_idx, train_idx = perm[:n_test], perm[n_test:]

X_train = torch.tensor(X[train_idx], dtype=torch.float32)
y_train = torch.tensor(y[train_idx])
X_test = torch.tensor(X[test_idx], dtype=torch.float32)
y_test = torch.tensor(y[test_idx])

# Architecture {2, 16, 8, 1}: ReLU hidden layers, sigmoid output, BCE loss, Adam.
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 1), nn.Sigmoid(),
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS, BATCH = 300, 32
start = time.perf_counter()
for epoch in range(EPOCHS):
    order = torch.randperm(len(X_train))
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        optimizer.zero_grad()
        loss = criterion(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
elapsed_ms = (time.perf_counter() - start) * 1000

with torch.no_grad():
    train_acc = ((model(X_train) >= 0.5).float() == y_train).float().mean().item()
    test_acc = ((model(X_test) >= 0.5).float() == y_test).float().mean().item()

print(f"training time : {elapsed_ms:.0f} ms")
print(f"train accuracy: {train_acc * 100:.2f}%")
print(f"test accuracy : {test_acc * 100:.2f}%")
