import logging
import os
import numpy as np
import torch
import random
import datetime
import subprocess
import csv
from mmcv.runner import get_dist_info
import torch.nn as nn
np.seterr(divide='ignore', invalid='ignore')

import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info

class _CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError

def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

def get_root_logger(log_level=logging.INFO, log_dir='./'):
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    logname = os.path.join(log_dir, thetime + '.log')
    logger = logging.getLogger()

    if not logger.hasHandlers():
        fmt ='%(asctime)s - %(levelname)s - %(message)s'
        format_str = logging.Formatter(fmt)
        logging.basicConfig(filename=logname, filemode='a', format=fmt, level=log_level)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        logger.addHandler(sh)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
    return logger

def logger_info(logger, dist, info):
    # to only write on rank0
    if not dist:
        logger.info(info)
    else:
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            logger.info(info)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_default_configs(cfgs, display=False):
    "set default configs for new coming features"
    default_dict = dict(
        lr_decline=(60, 75, 90),
        resume_epoch=0,
        load_from=None,
        log_level='INFO',
        white_box_attack=True,
        source_model_path=None,
        target_model_path=None,
        freeze_set=(),
        sampler=None,
        lr_dict=None,
        backbone="WideResNet",
        classifier="FC",
        feature_ext=False,
        centroids=False,
        begin_save=40,
        eval_rob=False,
        denoise=(),
        existing_ratio=1,
        imbalance_ratio=0.,
        beta=1.0,
        step_bar=2.0,
        method='',
        load_modules=(),
        targeted=False,
        remark='',
        alpha=1.0,
        margin_opt=None,
        target_opt=None,
        adv_loss_opt=None,
        nat_loss_opt=None,
        activation='relu',
        cpu_data=False,
        eval_freq=5,
        free_bn=False,
        other_params=dict(),
        load_model=None,
        deffer_opt=None,
        warmup_epochs=0,

    )

    # asserts for some setting
    assert cfgs.test_step_size == 0.0078

    print("[Default Configs]: ")
    for k, v in default_dict.items():
        if k not in cfgs:
            setattr(cfgs, k, v)
            print(" {} : {}".format(k, v), end=' | ')
    print()

def load_checkpoint(model, logger, filename, load_modules = (), strict=False, display=True, rename=False, check_module=True):

    logger.info('Loading checkpoint from %s', filename)
    if load_modules:
        logger.info('Ignoring module {}'.format(' '.join(load_modules)))
    if getattr(model, 'load_model', None):
        model.load_model(filename)

    if not rename and not check_module:
        model.load_state_dict(torch.load(filename), strict=strict)
        return

    if hasattr(model, 'model'):
        module = model.model
    else:
        module = model
    if hasattr(module, 'module'):
        module = module.module
    else:
        module = module
    own_state = module.state_dict()

    if rename:
        own_state_rename = dict()
        for name, param in own_state.items():
            prefix = name.split('.')[0]
            if prefix is 'module':
                name = '.'.join(name.split('.')[1:])
            prefix = name.split('.')[0]
            if prefix in ['backbone', 'classifier']:
                origin = '.'.join(name.split('.')[1:])
            else:
                origin = name
            own_state_rename.update({origin:param})
    else:
        own_state_rename = own_state
    unexpected_keys = []
    ignored_keys = []
    state_dict = torch.load(filename)
    state_dict_rename = dict()
    for name, param in state_dict.items():
        if name.split('.')[0] == 'model':
            name = '.'.join(name.split('.')[1:])
        if name.split('.')[0] == 'module':
            name = '.'.join(name.split('.')[1:])
        state_dict_rename.update({name:param})
        if load_modules:
            flag = 0
            for load_module in load_modules:
                if load_module in name:
                    flag = 1
        else:
            flag=1
        if not flag:
            ignored_keys.append(name)
            continue
        if name not in own_state_rename:
            if 'sub_block1' in name:
                continue
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state_rename[name].copy_(param)
            # print(own_state_rename[name][0,0,0])
            # print(own_state[name][0,0,0])
        except Exception:
            raise RuntimeError(
                'While copying the parameter named {}, '
                'whose dimensions in the model are {} and '
                'whose dimensions in the checkpoint are {}.'.format(
                    name, own_state[name].size(), param.size()))
    missing_keys = set(own_state_rename.keys()) - set(state_dict_rename.keys())
    err_msg = []
    if unexpected_keys:
        err_msg.append('>>> unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('>>> missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    if ignored_keys:
        err_msg.append('>>> mismatched / ignored key in source state_dict: {}\n'.format(
            ', '.join(ignored_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif display:
            if logger is not None:
                logger.warn(err_msg)
            else:
                print(err_msg)


if __name__ == '__main__':
    pass
