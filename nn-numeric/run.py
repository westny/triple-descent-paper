import os
import time
import subprocess
import itertools
import collections
import argparse
import torch
import numpy as np

from utils import copy_py, who_am_i


def create_script(params):
    script = '''#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --output={name}.out
#SBATCH --job-name={name}
#SBATCH --cpus-per-task=1
ulimit -n 64000

python main.py --name {name} --epochs {epochs} --noise {noise} --n {n} --width {width} --num_seeds {num_seeds} --lr {lr} --d {d} --test_noise {test_noise} --loss_type {loss_type} --n_classes 1 --task regression --no_cuda False --depth {depth} --wd {wd} --activation {activation} --dataset {dataset}
'''.format(**params)
    with open('{}.sbatch'.format(params['name']), 'w') as f:
        f.write(script)
    # with open('{}.params'.format(params['name']), 'wb') as f:
    #     torch.save(params, f)


def create_script_simple(params):
    #script = '''python main.py --name {name} --epochs {epochs} --noise {noise} --n {n} --batch-size {n} --width {width} --num_seeds {num_seeds} --lr {lr} --d {d} --test_noise {test_noise} --loss_type {loss_type} --n_classes 1 --task regression --depth {depth} --wd {wd} --activation {activation} --dataset {dataset}
    script = '''python main.py --name {name} --epochs {epochs} --noise {noise} --n {n} --batch-size 32 --width {width} --num_seeds {num_seeds} --lr {lr} --d {d} --loss_type {loss_type} --n_classes 1 --task regression --depth {depth} --wd {wd} --activation {activation} --dataset {dataset}
'''.format(**params)
    with open('{}.sh'.format(params['name']), 'w') as f:
        f.write(script)


def send_script(file):
    process = subprocess.Popen(['sbatch', file], stdout=subprocess.PIPE)


if __name__ == '__main__':

    exp_dir = 'r.{}'.format(int(time.time()))
    os.mkdir(exp_dir)
    copy_py(exp_dir)
    os.chdir(exp_dir)

    widths = np.unique(np.logspace(0, 2.5, 20).astype(int))
    ns = np.logspace(1, 5, 20).astype(int)

    #widths = np.unique(np.logspace(4, 8, 13, base=2).astype(int))
    #ns = np.logspace(5, 17, 13, base=2).astype(int)
    
    grid = collections.OrderedDict({
        'width': widths,
        'n': ns,
        'depth': [1],
        'wd': [0.],
        'activation': ['relu'],
        'dataset': ['random'],
        'noise': [0.1],
        'lr': [0.01],
        'd': [14 * 14],
        'num_seeds': [1],
        'test_noise': [False],
        'loss_type': ['mse'],
        'epochs': [1000],
    })


    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))


    for i, params in enumerate(dict_product(grid)):
        torch.save(grid, 'params.pkl')
        params['name'] = str(i)
        #create_script(params)
        create_script_simple(params)
        #file_name = '{}.sbatch'.format(params['name'])
        #send_script(file_name)
