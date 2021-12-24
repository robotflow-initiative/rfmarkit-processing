import torch
from IMUDataset import IMUDataset
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from tilo_network import get_model, get_loss
import tqdm
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

torch.random.manual_seed(0)
np.random.seed(0)

import os
class CoolSystem(pl.LightningModule):

    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()

        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def do_train(model, optim, train_set, eval_set):
    global MODEL_CONFIG, ACCELERATOR, DEVICE
    data = DataLoader(train_set, batch_size=MODEL_CONFIG['batch_sz'], shuffle=True, pin_memory=True)
    model, optim, data = ACCELERATOR.prepare(model, optim, data)
    train_step = 0

    for epoch in range(MODEL_CONFIG['n_epoch']):
        model.train()
        with tqdm.tqdm(range(len(data)), disable=not ACCELERATOR.is_main_process) as pbar:
            for stimulis, label in data:
                stimulis = stimulis.to(DEVICE)
                label = label.to(DEVICE)

                optim.zero_grad()
                pred, pred_cov = model(stimulis)
                loss = get_loss(pred, pred_cov, label, epoch)
                loss = torch.mean(loss)
                ACCELERATOR.backward(loss)
                optim.step()

                if ACCELERATOR.is_main_process:
                    WRITER.add_scalar('train_loss', float(loss.detach().cpu().numpy()), train_step)
                    train_step += 1

                pbar.set_description(f"train_epoch={epoch}, loss={loss.detach().cpu().numpy()}")
                pbar.update()

        ACCELERATOR.wait_for_everyone()
        attr = do_eval(model, eval_set, epoch)
        if ACCELERATOR.is_main_process:
            do_log(model, attr)
        ACCELERATOR.wait_for_everyone()
        ACCELERATOR.save(ACCELERATOR.unwrap_model(model), os.path.join(MODEL_CONFIG['save_dir'], f'epoch={epoch}.pth'))


def do_log(model, attr, epoch):
    model_unwarpped = ACCELERATOR.unwrap_model(model)
    WRITER.add_scalar('eval_loss', attr['losses'].mean(), epoch)
    for name, param in model_unwarpped.named_parameters():
        if 'bn' not in name:
            WRITER.add_histogram('model_layer' + name, param, epoch)


def do_eval(model, eval_set, epoch):
    global DEVICE, ACCELERATOR
    model.eval()
    data = DataLoader(eval_set, batch_size=MODEL_CONFIG['batch_sz'])
    data = ACCELERATOR.prepare(data)

    labels_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    with tqdm.tqdm(range(len(data)), disable=not ACCELERATOR.is_main_process) as pbar:
        for stimulis, label in data:
            stimulis = stimulis.to(DEVICE)
            label = label.to(DEVICE)
            pred, pred_cov = model(stimulis)
            loss = get_loss(pred, pred_cov, label, epoch)

            labels_all.append(label.detach().cpu().numpy())
            preds_all.append(pred.detach().cpu().numpy())
            preds_cov_all.append(pred_cov.detach().cpu().numpy())
            losses_all.append(loss.detach().cpu().numpy())
            pbar.update()

    labels_all = np.concatenate(labels_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)

    return {'labels': labels_all, 'preds': preds_all, 'preds_cov': preds_cov_all, 'losses': losses_all}


if __name__ == '__main__':
    MODEL_ARCH = 'resnet'
    MODEL_CONFIG = {
        'in_dim': 7,
        'input_dim': 9,
        'output_dim': 3,
        'features': ['acc', 'gyro', 'mag'],
        'batch_sz': 32,
        'n_epoch': 20,
        'lr': 1e-4,
        'check_point_dir': './checkpoint',
        'log_dir': './log'
    }
    

    ACCELERATOR = Accelerator()
    DEVICE = ACCELERATOR.device
    WRITER = SummaryWriter(MODEL_CONFIG['log_dir'])

    model = get_model(MODEL_ARCH, MODEL_CONFIG['in_dim'], MODEL_CONFIG['input_dim'], MODEL_CONFIG['output_dim'])
    optim = Adam(model.parameters(), lr=MODEL_CONFIG['lr'], weight_decay=0.)

    dataset = IMUDataset('./data_interp', features=MODEL_CONFIG['features'])
    train_len = int(len(dataset) * 0.001)
    eval_len = int(len(dataset) * 0.001)
    test_len = len(dataset) - eval_len - train_len
    train_set, eval_set, test_set = (Subset(dataset, range(train_len)), Subset(dataset, range(train_len, train_len + eval_len)),
                                     Subset(dataset, range(train_len + eval_len, len(dataset))))

    do_train(model, optim, train_set, eval_set)
