import torch
import torch.nn.functional as F
from IMUDataset import IMUDataset
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

model = torch.nn.Transformer().to(device)
optim = torch.optim.Adam(model.parameters())

dataset = load_dataset('my_dataset')
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optim, data = accelerator.prepare(model, optim, data)

model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.cross_entropy(oÎutput, targets)

        accelerator.backward(loss)
        optimizer.step()