#!/usr/bin/env python3

import os
import subprocess
import itertools
import psutil
import time
import sys

# Change to parent directory
os.chdir("../")

# Define parameters
algorithms = ['fedsam']
# algorithms = ['fedavg', 'fedavgm', 'feddyn', 'fedprox', 'scaffold', 'fedsam', 'fedasam', 'mofedsam', 'fedsmoo_noreg', 'fedgf']

datasets = ['cifar10']
# datasets = ['cifar10', 'cifar100']

models = ['FedSAMcnn']
wandb_project_name = "cifar"

sample_ratios = ['0.2']
# sample_ratios = ['0.05', '0.1', '0.2']

batch_sizes = ['64']
epochs = ['1']
lrs = ['0.01']
wd = '0.0004'

gpu_start = 0
gpu_end = 8
gpu = gpu_start

num_cpu = 9

# Fixed parameters
rhos = ['0.02']
c_os = ['0.2']


# Directory alphas will be set based on dataset
def set_param(dataset):
    if dataset == "cifar10":
        dir_alphas = ["0.05"]
        eval_every = 800
        round_ = 10000
    elif dataset == "cifar100":
        dir_alphas = ["0.5"]
        eval_every = 1000
        round_ = 20000
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dir_alphas, eval_every, round_


# Function to get the next GPU
def get_next_gpu(current_gpu):
    next_gpu = current_gpu + 1
    if next_gpu > gpu_end:
        next_gpu = gpu_start
    return next_gpu


# Function to set CPU affinity for a process
def set_cpu_affinity_windows(process, start_cpu, num_cpu_cores):
    try:
        p = psutil.Process(process.pid)
        cpu_range = list(range(start_cpu, start_cpu + num_cpu_cores))
        p.cpu_affinity(cpu_range)
    except psutil.AccessDenied:
        print(f"Access Denied when setting CPU affinity for PID {process.pid}")
    except Exception as e:
        print(f"Error setting CPU affinity for PID {process.pid}: {e}")


# Function to build and run the command
def run_command(cmd, gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Start the subprocess
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    # Set CPU affinity
    set_cpu_affinity_windows(process, gpu_id * num_cpu, num_cpu)

    return process


# Functions to handle different algorithms
def run_fedavg(params):
    cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
          f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
          f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
          f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} " \
          f"--dir_alpha {params['dir_alpha']} --transform --save_model"
    run_command(cmd, params['gpu'])


def run_fedavgm(params):
    betas = ["0.1", "0.7", "0.9"]  # Define betas as needed
    for beta in betas:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --beta {beta} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_scaffold(params):
    g_lrs = ['0.01', '0.001']
    for g_lr in g_lrs:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --g_lr {g_lr} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_fedprox(params):
    mus = ['0.01', '0.001']
    for mu in mus:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --mu {mu} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_feddyn(params):
    alphas = ['0.1', '0.01', '0.001']
    for alpha in alphas:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --alpha {alpha} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_fedsam(params):
    dataset = params['dataset']
    dir_alpha = params['dir_alpha']
    if dataset == "cifar10":
        if dir_alpha in ["0", "0.05"]:
            rho = "0.1"
        elif dir_alpha == "100":
            rho = "0.02"
    elif dataset == "cifar100":
        if dir_alpha == "0":
            rho = "0.02"
        elif dir_alpha in ["0.5", "1000"]:
            rho = "0.05"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
          f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
          f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
          f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} " \
          f"--dir_alpha {params['dir_alpha']} --transform --save_model"
    run_command(cmd, params['gpu'])


def run_fedasam(params):
    dataset = params['dataset']
    dir_alpha = params['dir_alpha']
    eta = "0.2"  # Fixed as per your script

    if dataset == "cifar10":
        if dir_alpha in ["0", "0.05"]:
            rho = "0.7"
        elif dir_alpha == "100":
            rho = "0.05"
    elif dataset == "cifar100":
        rho = "0.5"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
          f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
          f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
          f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} " \
          f"--eta {eta} --dir_alpha {params['dir_alpha']} --transform --save_model"
    run_command(cmd, params['gpu'])


def run_mofedsam(params):
    betas = ["0.1", "0.7", "0.9"]
    for rho in rhos:
        for beta in betas:
            cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
                  f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
                  f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
                  f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} --beta {beta} " \
                  f"--dir_alpha {params['dir_alpha']} --transform --save_model"
            run_command(cmd, params['gpu'])


def run_fedgf(params):
    W = "30"
    for rho, c_o in itertools.product(rhos, c_os):
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} --c_o {c_o} " \
              f"--dir_alpha {params['dir_alpha']} --W {W} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_fedsmoo_noreg(params):
    for rho in rhos:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


def run_fedgamma(params):
    for rho in rhos:
        cmd = f"python -u main.py --eval_every {params['eval_every']} --wandb_project_name {params['wandb_project_name']} " \
              f"--model {params['model']} --alg {params['algorithm']} --dataset {params['dataset']} --total_client 100 " \
              f"--com_round {params['round']} --sample_ratio {params['sample_ratio']} --batch_size {params['batch_size']} " \
              f"--epochs {params['epoch']} --lr {params['lr']} --weight_decay {params['wd']} --rho {rho} " \
              f"--dir_alpha {params['dir_alpha']} --transform --save_model"
        run_command(cmd, params['gpu'])


# Mapping algorithms to their respective functions
algorithm_functions = {
    "fedavg": run_fedavg,
    "fedavgm": run_fedavgm,
    "scaffold": run_scaffold,
    "fedprox": run_fedprox,
    "feddyn": run_feddyn,
    "fedsam": run_fedsam,
    "fedasam": run_fedasam,
    "mofedsam": run_mofedsam,
    "fedgf": run_fedgf,
    "fedsmoo_noreg": run_fedsmoo_noreg,
    "fedgamma": run_fedgamma
}

# List to keep track of subprocesses
processes = []

try:
    for dataset in datasets:
        print("############################################## Running ##############################################")
        dir_alphas, eval_every, round_ = set_param(dataset)
        for dir_alpha in dir_alphas:
            for model in models:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for epoch in epochs:
                            for algorithm in algorithms:
                                for sample_ratio in sample_ratios:
                                    print(
                                        f"Launching process on GPU {gpu} with CPU cores {gpu * num_cpu}-{gpu * num_cpu + num_cpu - 1}")

                                    # Prepare parameters
                                    params = {
                                        'eval_every': eval_every,
                                        'wandb_project_name': wandb_project_name,
                                        'model': model,
                                        'algorithm': algorithm,
                                        'dataset': dataset,
                                        'round': round_,
                                        'sample_ratio': sample_ratio,
                                        'batch_size': batch_size,
                                        'epoch': epoch,
                                        'lr': lr,
                                        'wd': wd,
                                        'dir_alpha': dir_alpha,
                                        'gpu': gpu
                                    }

                                    # Get the function based on the algorithm
                                    func = algorithm_functions.get(algorithm)
                                    if func:
                                        func(params)
                                    else:
                                        print(f"Unknown algorithm: {algorithm}")

                                    # Update GPU for next process
                                    gpu = get_next_gpu(gpu)

                                    # Optional: Add delay to prevent overwhelming the system
                                    time.sleep(0.1)

    # Optionally, wait for all processes to complete
    # for proc in processes:
    #     proc.wait()

except KeyboardInterrupt:
    print("Terminating all subprocesses...")
    for proc in processes:
        proc.terminate()
    print("All subprocesses terminated.")
