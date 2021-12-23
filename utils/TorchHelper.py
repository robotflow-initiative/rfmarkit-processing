import torch
import torch.nn as nn
import pickle
from typing import List, Union, Dict, Optional
from .TorchContext import TorchContext
import os
import torch.multiprocessing as mp
import subprocess


class TorchHelper:
    def __init__(self):
        pass

    @classmethod
    def print_debug(cls, level: Union[int, str], msg):
        """Format a simple debug message
        Args:
            level: Type of information
            msg:

        Returns:

        """
        _levels = ('Info', 'Debug', 'Warning', 'Error')
        if isinstance(level, int):
            try:
                level = _levels[level]
            except KeyError:
                level = 'Info'

        print("[ {} ] {}".format(level, msg))

    @classmethod
    def config_parallel(cls, model: nn.Module, config) -> Union[nn.Module, nn.DataParallel]:
        """
        Using DataParallel

        """
        print('[ Info ] Using DataParallel')
        available_device_idxs = config.available_devices
        print('[ Info ] Available devices: {}'.format(available_device_idxs))
        # Set variable to "0,1,2,3" for example
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in list(range(torch.cuda.device_count()))])

        if config.use_cuda:
            if len(available_device_idxs) > 0:
                torch.cuda.set_device(available_device_idxs[0])  # to the first free device
                model = torch.nn.DataParallel(model.cuda(), device_ids=available_device_idxs)
            else:
                print('[ Warning ] No available_device')
        else:
            model = torch.nn.DataParallel(model)

        return model

    @classmethod
    def load_from_DataParallel(cls, model: nn.Module, path_to_checkpoint: str, **kwargs) -> nn.Module:
        if not isinstance(model, nn.DataParallel):
            model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(path_to_checkpoint, **kwargs))
        model = model.module  # Cancel DataParallel
        return model

    @classmethod
    def load_from_PlainModel(cls, model: nn.Module, path_to_checkpoint: str) -> nn.Module:
        model.load_state_dict(torch.load(path_to_checkpoint))
        return model

    @classmethod
    def save_model_state_dict(cls, model: nn.Module, path: str):
        torch.save(model.state_dict(), path)

    @classmethod
    def load_checkpoint_pkl(cls, path_to_pkl: str) -> dict:
        """load a custom pkl

        Assumption:
            The pkl file contains following keys:
            - "config":  TorchConfig object for train configuration
            - "context": TorchContext object of the record
            - "state_dict": model parameters
            - "appendix": Dict[str, str]: a bundle of appended files to record model structure / train workflow

        Args:
            path_to_pkl:

        Returns:
        """
        with open(path_to_pkl, 'rb') as f:
            obj: dict = pickle.load(f)
        assert isinstance(obj, dict)
        assert 'config' in obj.keys()
        assert 'context' in obj.keys()
        assert 'state_dict' in obj.keys()
        assert 'appendix' in obj.keys()
        return obj

    @classmethod
    def dump_checkpoint_pkl(cls, path_to_pkl: str, config, context: TorchContext, state_dict, appendix: List[str]):
        """

        Args:
            appendix:
            state_dict:
            path_to_pkl: Save path
            config: TorchConfig object for train configuration
            context:  TorchContext object of the record


        Returns:

        """
        os.mknod(path_to_pkl)
        checkpoint = dict()
        checkpoint['config'] = config
        checkpoint['context'] = context
        checkpoint['state_dict'] = state_dict
        checkpoint['appendix'] = dict()

        with open(path_to_pkl, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, dict)
        assert 'config' in obj.keys()
        assert 'context' in obj.keys()
        assert 'appendix' in obj.keys()

        for item in appendix:
            with open(item, 'r') as f:
                checkpoint['appendix'][item] = ''.join(f.readlines())

        with open(path_to_pkl, 'wb') as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def dump_appendix(cls, appendix: List[str]) -> Dict[str, str]:
        out = dict()
        for item in appendix:
            with open(item, 'r') as f:
                out[item] = ''.join(f.readlines())
        return out

    @classmethod
    def launch_ddp_single_node(cls, f, ddp_cfg: dict, train_cfg, ):
        train_cfg['use_ddp'] = True  # Force this flag to be true
        train_cfg['ddp_init_method'] = ddp_cfg['init_method']
        train_cfg['ddp_world_size'] = len(train_cfg.available_devices)

        mp.spawn(f, args=(train_cfg,), nprocs=len(train_cfg.available_devices))

    @staticmethod
    def get_device_by_idx(idx: int = 0):
        """Return a torch.device instance
        Args:
            idx:

        Returns:

        """
        return torch.device('cuda:{}'.format(idx) if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_device_free_memory() -> List[int]:
        """Use nvidia-smi to probe gpu memory usage

        Returns:
            A list of integers of device memories
        """

        # Use grep to extract message
        p = subprocess.Popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free', shell=True, stdout=subprocess.PIPE)
        p.wait()
        output: List[bytes] = p.stdout.readlines()
        memory: List[int] = [int(x.split()[2]) for x in output]
        return memory
