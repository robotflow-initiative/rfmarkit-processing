import json
from typing import List, Tuple, Optional
import argparse
import os
import torch
import platform
from torch.utils.data import Dataset
from .TorchHelper import TorchHelper


class TorchConfig:
    """A configuration for PyTorch

    Properties:
        current_dir:[str] Current directory from os module

        num_epoch:  [int] Number of epoch
        batch_sz:   [int] Batch size
        lr:         [float] Learning rate

        ## Workflow related
        mode:               [List[str,...]] Mode of working, can be combinitions of 'train'|'test'
        save_model:         [bool]  Save modules or not. Default to False
        save_interval:      [int]   Interval to save modules
        optim_interval:     [int]   Optimize after n batches Default to 1
        eval_interval:      [int]   Interval to eval modules
        model_output_dir:   [str]   Directory for trained model_weights. Default to './model_output/'
        log_dir:            [str]   Directory for logs. Default to './log/'
        segmentations:      [Tuple[float,float,float]] Segmenting datasets. Range [0-1] Default to [0.6,0.2,0.2]
        info:               [str]   Optional information. Default to 'Hello world!'
        fine_tune:          [bool]  If we want to fine tune a modules. Default to false
        model_input_path:   [str]   modules path for fine_tune
        validation:         [str]   'cross' or 'static'

        # CUDA related
        dtype:              [str]   type of modules weights. Default to torch.float64
        use_cuda:           [bool]  Use cuda or not, Default to True
        num_workers:        [int]   Number of workers to load data
        use_parallel:       [bool]  Use parallel training or not
        use_ddp:            [bool]  Use distributed parallel training or not
        memory_thresh       [float|int] Minimum free memory on GPU, default to 4096 (MiB)
        available_devices   [List[int]] Manually set available devices
        default_device      [int]   Default device when DataParallel is not configured

        # DistributedDataParallel
        ddp_init_method     [int]   init_method for DDP, default to 28888
        ddp_world_size      [int]   number of nodes
        ddp_rank            [int]   rank of current node
        ddp_local_rank      [int]   rank of GPU


        optimizer:          [str]   Optimizer[optional]
        loss:               [str]   Loss[optional]
        dropout:            [bool]  Dropout or not[optional]
        dropout_rate:       [float] Dropout rate[optional]

    """
    def __init__(self, path_to_config: Optional[str] = None):
        if path_to_config is not None:
            with open(path_to_config, encoding='utf-8') as f:
                self.hyper_parameters: dict = json.load(f)
        else:
            self.hyper_parameters: dict = dict()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self._verify_config()

    def __repr__(self):
        string = ""
        string += "TorchConfig[num_epoch={}".format(self.num_epoch)
        string += "            batch_sz={}".format(self.batch_sz)
        string += "            lr={}".format(self.lr)
        string += "            info={}".format(self.info)
        string += "            fine_tune={}".format(self.fine_tune)
        string += "           ]"
        return string

    def __setitem__(self, key, value):
        self.hyper_parameters[key] = value

    def __getitem__(self, item):
        if item in self.hyper_parameters.keys():
            return self.hyper_parameters[item]
        else:
            return None

    def _verify_config(self):
        assert not all([self.use_ddp, self.use_parallel])

    def split_dataset(self, whole_dataset) -> List[torch.utils.data.Subset]:
        """ A datasets can be spilt to subset using this function
        Args:
            whole_dataset:
        Returns:

        """
        tot_len = len(whole_dataset)
        train_seg = int(self.train_segmentation * tot_len)
        eval_seg = int(self.eval_segmentation * tot_len)
        test_seg = tot_len - train_seg - eval_seg

        return torch.utils.data.random_split(whole_dataset, (train_seg, eval_seg, test_seg))

    @staticmethod
    def parse_args():
        """This method can parse commandline arguments
        Returns:
            parsed arguments
        Todo:
            finish this parser
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parser = argparse.ArgumentParser(description="Train a network that recognize handwritten numbers")
        parser.add_argument("--num_epoch", default=10, type=int, help="set epoch nums")
        parser.add_argument("--lr", default=1e-3, type=float, help="set learning rate")
        parser.add_argument("--batch_sz", default=32, type=int, help="set batch_size")
        parser.add_argument("--model_dir",
                            default=os.path.join(current_dir, "models"),
                            type=str,
                            help="set the directory to save mdoel")
        parser.add_argument("--model_input_path", type=str, help="the path to the modules to be loaded")
        parser.add_argument("--log_dir", default=os.path.join(current_dir, "log"), type=str, help="directory to store logs")
        parser.add_argument("--fine_tune", action="store_true", dest="fine_tunning", help="fine tuning a modules")
        parser.add_argument("--save_model", action="store_true", dest="save_model", help="saved the trained modules")

        args = parser.parse_args()

        return args

    @property
    def num_epoch(self) -> int:
        if 'num_epoch' in self.hyper_parameters.keys():
            return int(self.hyper_parameters['num_epoch'])
        else:
            return 0

    @property
    def batch_sz(self) -> int:
        if 'batch_sz' in self.hyper_parameters.keys():
            return int(self.hyper_parameters['batch_sz'])
        else:
            return 1

    @property
    def lr(self) -> float:
        if 'lr' in self.hyper_parameters.keys():
            return self.hyper_parameters['lr']
        else:
            return 0

    @property
    def mode(self) -> List[str]:
        if 'mode' in self.hyper_parameters.keys():
            return self.hyper_parameters['mode']
        else:
            return ['train', 'debug']

    @property
    def save_model(self) -> bool:
        if 'save_model' in self.hyper_parameters.keys():
            return self.hyper_parameters['save_model']
        else:
            return False

    @property
    def save_interval(self) -> int:
        if 'save_interval' in self.hyper_parameters.keys():
            return self.hyper_parameters['save_interval']
        else:
            return 100

    @property
    def optim_interval(self) -> int:
        if 'optim_interval' in self.hyper_parameters.keys():
            return self.hyper_parameters['optim_interval']
        else:
            return 1

    @property
    def eval_interval(self) -> int:
        if 'eval_interval' in self.hyper_parameters.keys():
            return self.hyper_parameters['eval_interval']
        else:
            return 10

    @property
    def model_output_dir(self) -> str:
        if 'model_output_dir' in self.hyper_parameters.keys():
            return self.hyper_parameters['model_output_dir']
        else:
            return "./model_output/"

    @property
    def log_dir(self) -> str:
        if 'log_dir' in self.hyper_parameters.keys():
            return self.hyper_parameters['log_dir']
        else:
            return "./log/"

    @property
    def train_segmentation(self) -> float:
        try:
            return self.hyper_parameters['segmentation']['train']
        except ValueError or KeyError:
            return 0.6

    @property
    def eval_segmentation(self) -> float:
        try:
            return self.hyper_parameters['segmentation']['eval']
        except ValueError or KeyError:
            return 0.2

    @property
    def test_segmentation(self) -> float:
        try:
            return self.hyper_parameters['segmentation']['eval']
        except ValueError or KeyError:
            return 0.2

    @property
    def info(self) -> str:
        if 'info' in self.hyper_parameters.keys():
            return self.hyper_parameters['info']
        else:
            return "Hello world!"

    @property
    def fine_tune(self) -> bool:
        if 'fine_tune' in self.hyper_parameters.keys():
            return self.hyper_parameters['fine_tune']
        else:
            return False

    @property
    def model_input_path(self) -> str:
        if 'model_input_path' in self.hyper_parameters.keys():
            return self.hyper_parameters['model_input_path']
        else:
            return ""

    @property
    def dtype(self) -> torch.dtype:
        if 'dtype' in self.hyper_parameters.keys():
            if self.hyper_parameters['dtype'] == 'float16':
                return torch.float16
            if self.hyper_parameters['dtype'] == 'float32':
                return torch.float32
            if self.hyper_parameters['dtype'] == 'float64':
                return torch.float64
        else:
            return torch.float32

    @property
    def use_cuda(self) -> bool:
        if 'use_cuda' in self.hyper_parameters.keys():
            return self.hyper_parameters['use_cuda']
        else:
            return True

    @property
    def num_workers(self) -> int:
        # In docker and windows num_worke=0
        if os.path.exists('/.dockerenv') or 'Windows' in platform.platform() or 'debug' in self.mode:
            return 0

        if 'num_workers' in self.hyper_parameters.keys():
            return self.hyper_parameters['num_workers']
        else:
            return 0

    @property
    def use_parallel(self) -> bool:
        if 'use_parallel' in self.hyper_parameters.keys():
            return self.hyper_parameters['use_parallel']
        else:
            return False

    @property
    def use_ddp(self) -> bool:
        if 'use_ddp' in self.hyper_parameters.keys():
            return self.hyper_parameters['use_ddp']
        else:
            return False

    @property
    def memory_thresh(self) -> int:
        # In docker and windows num_worke=0
        if 'memory_thresh' in self.hyper_parameters.keys():
            return self.hyper_parameters['memory_thresh']
        else:
            return 4096

    @property
    def available_devices(self) -> List[int]:
        """Return a list of available devices

        Returns:
            A list of devices.For example, [0, 1, 2]
        """
        if 'available_devices' in self.hyper_parameters.keys():
            _devices = self.hyper_parameters['available_devices']
            return list(map(lambda x: int(x), _devices))
        else:
            self.hyper_parameters['available_devices'] = [
                idx for idx, mem in enumerate(TorchHelper.get_device_free_memory()) if mem > self.memory_thresh
            ]
            return self.hyper_parameters['available_devices']

    @property
    def default_device(self):
        if 'default_device' in self.hyper_parameters.keys():
            return self.hyper_parameters['default_device']
        else:
            return self.available_devices[0]

    @property
    def ddp_init_method(self) -> str:
        if 'ddp_init_method' in self.hyper_parameters.keys():
            return self.hyper_parameters['ddp_init_method']
        else:
            return 'tcp://localhost:28888'

    @property
    def ddp_world_size(self) -> int:
        if 'ddp_world_size' in self.hyper_parameters.keys():
            return self.hyper_parameters['ddp_world_size']
        else:
            return 1

    @property
    def ddp_rank(self) -> int:
        if 'ddp_rank' in self.hyper_parameters.keys():
            return self.hyper_parameters['ddp_rank']
        else:
            return 0

    @property
    def ddp_local_rank(self) -> Optional[int]:
        if not self.use_cuda:
            return None

        if 'ddp_local_rank' in self.hyper_parameters.keys():
            return self.hyper_parameters['ddp_local_rank']
        else:
            try:
                parser = argparse.ArgumentParser()
                parser.add_argument("--local_rank", type=int)
                args = parser.parse_args()
                self.hyper_parameters['ddp_local_rank'] = args.local_rank
            except AttributeError:
                self.hyper_parameters['ddp_local_rank'] = None

            return self.hyper_parameters['ddp_local_rank']

    @property
    def ddp_local_gpu(self) -> Optional[int]:
        if not self.use_cuda:
            return None

        if 'ddp_local_gpu' in self.hyper_parameters.keys():
            return self.hyper_parameters['ddp_local_gpu']
        else:
            return 0

    @property
    def appendix(self) -> List[Optional[str]]:
        if 'appendix' in self.hyper_parameters.keys():
            return self.hyper_parameters['appendix']
        else:
            return []

    @classmethod
    def generate_template(cls, path_to_config: str):
        TEMPLATE = """{
  "num_epoch": 256,
  "batch_sz": 3,
  "lr": 0.001,
  "save_model": true,
  "save_interval": 64,
  "optim_interval": 2,
  "eval_interval": 64,
  "fine_tune": true,
  "model_input_path": "./model_output/plain_baseline.pth",
  "num_workers":2,
  "use_parallel": true,
  "use_ddp": false,
  "available_devices": [
    0,
    2,
    3
  ],
  "default_device": 3,
  "mode": [
    "train","debug"
  ],
  "info": "train precoder with encoder loss",
  "train_len": 4,
  "eval_len": 16
}
"""
        with open(path_to_config, 'w') as f:
            f.write(TEMPLATE)
