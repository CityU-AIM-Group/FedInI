import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Conv2d(3, 64, 3)
optimizer = optim.SGD(model.parameters(), lr=0.5)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

for i in range(5):
    optimizer.zero_grad()
    print(i)
    print("xxxxxxxxxxxxxxx")
    print(model.weight[0, ...])
    x = model(torch.randn(3, 3, 64, 64))
    loss = x.sum()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    print("xxxxxxxxxxxxxxx")
    print(model.weight[0, ...])
    optimizer.step()
    print("xxxxxxxxxxxxxxx")
    print(model.weight[0, ...])