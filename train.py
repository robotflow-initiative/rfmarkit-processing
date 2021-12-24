import torch
import torch.nn.functional as F
from IMUDataset import IMUDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tilo_network import get_model, get_loss
import tqdm

MODEL_ARCH = 'resnet'
MODEL_CONFIG = {
    'in_dim': 7,
    'input_dim': 9,
    'output_dim': 3,
    'features': ['acc', 'gyro', 'mag'],
    'batch_sz': 32,
    'n_epoch': 20,
    'lr': 1e-4,
    'check_point_dir':'./checkpoint'
}
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

model = get_model(MODEL_ARCH, MODEL_CONFIG['in_dim'], MODEL_CONFIG['input_dim'], MODEL_CONFIG['output_dim'])
optim = Adam(model.parameters(), lr=MODEL_CONFIG['lr'], weight_decay=0.)

dataset = IMUDataset('./data_interp', features=MODEL_CONFIG['features'])
data = DataLoader(dataset, batch_size=MODEL_CONFIG['batch_sz'], shuffle=True, pin_memory=True)

model, optim, data = accelerator.prepare(model, optim, data)

model.train()
for epoch in range(MODEL_CONFIG['n_epoch']):
    with tqdm.tqdm(range(len(data))) as pbar:
        for stimulis, label in data:
            stimulis = stimulis.to(device)
            label = label.to(device)

            optim.zero_grad()

            pred, pred_cov = model(stimulis)
            loss = get_loss(pred, pred_cov, label, epoch)
            loss = torch.mean(loss)

            accelerator.backward(loss)
            optim.step()
            pbar.set_description(f"epoch={epoch}")
            pbar.update()
    