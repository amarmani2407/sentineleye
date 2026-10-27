"""Step 6b: teach the autoencoder what 'normal' looks like."""
import json
import numpy as np
import torch
from torch import nn
from anomaly_model import AE, batch_scores, MODEL_FILE, THRESHOLD_FILE

EPOCHS = 40
BATCH = 64
SAFETY_MARGIN = 1.25     # threshold = 99.5th percentile of normal scores x this

torch.manual_seed(0)
np.random.seed(0)

X = np.load("normal_frames.npy").astype(np.float32) / 255.0
print(f"Loaded {len(X)} normal pictures.")
if len(X) < 200:
    print("Warning: very little data. Results will be poor. Record more with collect_normal.py")

# Train on the first 80%, keep the LAST 20% aside to measure what 'normal' scores look like
split = int(len(X) * 0.8)
train, val = X[:split], X[split:]

model = AE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    order = np.random.permutation(len(train))
    total = 0.0
    for i in range(0, len(train), BATCH):
        idx = order[i:i + BATCH]
        x = torch.from_numpy(train[idx]).unsqueeze(1)
        # Random brightness change so normal lighting shifts do not cause false alarms
        gain = torch.empty(len(idx), 1, 1, 1).uniform_(0.85, 1.15)
        x = (x * gain).clamp(0, 1)
        loss = loss_fn(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss {total / len(train):.5f}")

val_scores = batch_scores(model, val)
p995 = float(np.percentile(val_scores, 99.5))
threshold = p995 * SAFETY_MARGIN

print("\nNormal scores on the held-out pictures:")
print(f"  average {val_scores.mean():.5f} | 99.5th percentile {p995:.5f} | max {val_scores.max():.5f}")
print(f"  alert threshold = {threshold:.5f}")
print(f"  held-out pictures above threshold: {int((val_scores > threshold).sum())} of {len(val_scores)}")

torch.save(model.state_dict(), MODEL_FILE)
with open(THRESHOLD_FILE, "w") as f:
    json.dump({"threshold": threshold, "p995": p995, "mean": float(val_scores.mean())}, f)
print(f"\nSaved {MODEL_FILE} and {THRESHOLD_FILE}")
