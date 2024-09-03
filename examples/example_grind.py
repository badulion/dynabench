from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.model import NeuralPDE
from dynabench.model.grind import GrIND

import torch.optim as optim
import torch.nn as nn

download_equation('burgers', structure='cloud', resolution='low')

burgers_train_iterator = DynabenchIterator(split="train",
                                           equation='burgers',
                                           structure='cloud',
                                           resolution='low',
                                           lookback=1,
                                           rollout=1)

train_loader = DataLoader(burgers_train_iterator, batch_size=32, shuffle=True)

prediction_net = NeuralPDE(input_dim=2, hidden_channels=64, hidden_layers=3,
                solver={'method': 'euler', 'options': {'step_size': 0.1}},
                use_adjoint=False)

model = GrIND(prediction_net, num_ks=9, grid_resolution=64, spatial_dim=2)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for i, (x, y, p) in enumerate(train_loader):
        x, y, p = x[:,0].float(), y.float(), p.float() # only use the first channel and convert to float32
        optimizer.zero_grad()
        y_pred = model(x, p)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        break
    break

burgers_test_iterator = DynabenchIterator(split="test",
                                          equation='burgers',
                                          structure='cloud',
                                          resolution='low',
                                          lookback=1,
                                          rollout=16)

test_loader = DataLoader(burgers_test_iterator, batch_size=32, shuffle=False)

model.eval()

loss_values = []
for i, (x, y, p) in enumerate(test_loader):
    x, y, p = x[:,0].float(), y.float(), p.float() # only use the first channel and convert to float32
    y_pred = model(x, p, t_eval=range(17))
    loss = criterion(y_pred, y)
    loss_values.append(loss.item())

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")