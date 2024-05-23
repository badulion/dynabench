from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.model import NeuralPDE

import torch.optim as optim
import torch.nn as nn

download_equation('burgers', structure='grid', resolution='low')

burgers_train_iterator = DynabenchIterator(split="train",
                                           equation='burgers',
                                           structure='grid',
                                           resolution='low',
                                           lookback=1,
                                           rollout=1)

train_loader = DataLoader(burgers_train_iterator, batch_size=32, shuffle=True)

model = NeuralPDE(input_dim=2, hidden_channels=64, hidden_layers=3,
                solver={'method': 'euler', 'options': {'step_size': 0.1}},
                use_adjoint=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for i, (x, y, p) in enumerate(train_loader):
        x, y = x[:,0].float(), y.float() # only use the first channel and convert to float32
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

burgers_test_iterator = DynabenchIterator(split="test",
                                          equation='burgers',
                                          structure='grid',
                                          resolution='low',
                                          lookback=1,
                                          rollout=16)

test_loader = DataLoader(burgers_test_iterator, batch_size=32, shuffle=False)

model.eval()

loss_values = []
for i, (x, y, p) in enumerate(test_loader):
    x, y = x[:,0].float(), y.float() # only use the first channel and convert to float32
    y_pred = model(x, t_eval=range(17))
    loss = criterion(y_pred, y)
    loss_values.append(loss.item())

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")