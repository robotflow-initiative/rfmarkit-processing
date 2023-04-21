import torch
from IMUDataset import IMUDatasetACC2ACC
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from tilo_network.losses import loss_mse
import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_logger
from autoencoders import UNetAutoEncoder

torch.random.manual_seed(0)
np.random.seed(0)


class UNetAEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UNetAutoEncoder(in_channel=3, out_channel=3)
        self.automatic_optimization = False
        self.save_hyperparameters(self.config)
        self.epoch_idx = 0

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        stimulis, label = batch
        optim = self.optimizers(use_pl_optimizer=True)
        optim.zero_grad()
        pred = self.model(stimulis)
        loss = loss_mse(pred, label)
        loss = torch.mean(loss)
        self.manual_backward(loss)
        optim.step()
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.epoch_idx += 1
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                self.log('model_params' + name, param)

    def on_train_start(self) -> None:
        self.epoch_idx = 0

    def validation_step(self, batch, batch_idx):
        stimulis, label = batch
        pred = self(stimulis)
        loss = loss_mse(pred, label)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['val_loss'].mean() for x in outputs]).mean()
        self.log('val_avg_loss', avg_loss)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config['lr'], weight_decay=0.)


if __name__ == '__main__':
    MODEL_ARCH = 'resnet'
    MODEL_CONFIG = {
        'arch': MODEL_ARCH,
        'input_dim': 3,
        'output_dim': 3,
        'features': ['acc'],
        'batch_sz': 32,
        'n_epoch': 20,
        'lr': 1e-4,
        'check_point_dir': './checkpoint',
        'log_dir': './log',
        'num_workers': 1,
        'target': ['acc']
    }

    dataset = IMUDatasetACC2ACC('./data_interp_py900_1000')
    train_len = int(len(dataset) * 0.001)
    val_len = int(len(dataset) * 0.001)
    test_len = len(dataset) - val_len - train_len
    train_set, val_set, test_set = (Subset(dataset, range(train_len)), Subset(dataset, range(train_len, train_len + val_len)),
                                    Subset(dataset, range(train_len + val_len, len(dataset))))

    train_loader = DataLoader(train_set,
                              batch_size=MODEL_CONFIG['batch_sz'],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=MODEL_CONFIG['num_workers'])
    val_loader = DataLoader(val_set,
                            batch_size=MODEL_CONFIG['batch_sz'],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=MODEL_CONFIG['num_workers'])

    logger = pl_logger.TensorBoardLogger(save_dir=MODEL_CONFIG['log_dir'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="{epoch}-{val_loss:.4f}",
                                                       monitor='val_loss',
                                                       save_last=True,
                                                       save_top_k=20,
                                                       mode='min',
                                                       save_weights_only=False,
                                                       every_n_epochs=1,
                                                       save_on_train_epoch_end=True)

    trainer = pl.Trainer(max_epochs=25,
                         callbacks=[checkpoint_callback],
                         checkpoint_callback=True,
                         progress_bar_refresh_rate=10,
                         default_root_dir=MODEL_CONFIG['check_point_dir'],
                         logger=logger)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="{epoch}-{val_loss:.4f}",
                                                       monitor='val_loss',
                                                       save_last=True,
                                                       save_top_k=20,
                                                       mode='min',
                                                       save_weights_only=False,
                                                       every_n_epochs=1,
                                                       save_on_train_epoch_end=True)

    unetae = UNetAEModel(MODEL_CONFIG)
    trainer.fit(unetae, train_loader, val_loader)
