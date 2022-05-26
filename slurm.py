#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pathlib
import shutil

import submitit
import easydict

# from pytorch_experiments import (
from experiments import (
    algo_to_params,
    run_and_plot,
    ExplorationHparams,
    LinearModelHparams,
    NNParams,
)

METHODS=["AdversarialPLOT","PLOT"]
DATASETS=["Adult"]
VERSION_SUFFIX='v2'
TIMESTEPS=2000
NUM_EXPERIMENTS=1



WORKING_DIRECTORY = "/data/engs-oxfair/math1133/NEURIPS_exp"
PARTITION = "medium"
GPUS = 1
EXP_N = 1
PARALLEL = True
FAST = False
# Hardcoded
BATCH = 32
DECAY = 0.05
VERSION = f"_{TIMESTEPS}t_{'_'.join(METHODS)}_{'_'.join(DATASETS)}_{VERSION_SUFFIX}"
JOB_PREFIX = "exp_"
PARALLEL_STR = "_parallel" if PARALLEL else ""
JOB_NAME = f"{JOB_PREFIX}{PARALLEL_STR}_{VERSION}"
# Hardcoded
# T = 2000
# TODO is this also hardcoded now?
# EPS = 0.2

# Mahlanobis
# EPS_GREEDY = False
# GREEDY = False
# MAHL = True

# Eps
# EPS_GREEDY = True
# GREEDY = False
# MAHL = False

# Greed
# EPS_GREEDY = False
# GREEDY = True
# MAHL = False

# pseudo
# EPS_GREEDY = False
# GREEDY = False
# MAHL = False

# DECAY = 0.1
# DECAY = 0.0001
# TODO weight decay parameter???
# DECAY = 0.05

# TODO this is just used for naming???
# METHOD = "pseudolabel_"
# if EPS_GREEDY:
#     METHOD = f"eps_greedy_schedule_{EPS}_"
# if GREEDY:
#     METHOD = "greedy_bad_"
# if MAHL:
#     METHOD = "mahlanobis_alpha_4_rerun"
# VERSION = f"_{T}t_{METHOD}decay_{DECAY}_multi_exp_more_logs_v2"

def copy_and_run_with_config(
    run_fn, run_config, directory, parallel=False, additional_params={}, **cluster_config,
):
    print("Let's use slurm!")
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    # TODO clean-up
    ignore_list = [
        "checkpoints",
        "experiments",
        "experiment_results",
        "experiment_results_no_intercept",
        "experiment_results_no_intercept_std",
        ".git",
        "output",
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: ignore_list)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory)
#     print(additional_params)
#     executor.update_parameters(slurm_additional_parameters=additional_params)
    print(cluster_config)
    executor.update_parameters(**cluster_config)
    if parallel:
        print(run_config)
        jobs = executor.map_array(
            run_fn,
            # *args,
            *run_config
        )
        print(f"job_ids: {jobs}")
    else:
        job = executor.submit(run_fn, run_config)
        print(f"job_id: {job}")


def get_parallel_args(algos, datasets):
    training_modes = ["full_minimization"] * (len(datasets)*len(algos))
    args = easydict.EasyDict({
        "ray": False,
        "num_experiments_per_machine": 1,
        "T": TIMESTEPS,
        "baseline_steps":20_000,
        "batch_size": 32,
        "training_mode": "full_minimization",
        # Different in slurm, passed instead.
        # "datasets": DATASETS,
        # "algo_names": ALGOS
    })
    nn_params = NNParams()
    nn_params.max_num_steps = args.T
    nn_params.batch_size = args.batch_size
    nn_params.baseline_steps = args.baseline_steps
    nn_param_list = [nn_params] * (len(datasets)*len(algos))
    linear_model_hparams = [LinearModelHparams()] * (len(datasets)*len(algos))
    exploration_hparams = [algo_to_params(algo) for algo in algos]
    exploration_hparams = sum([[param] * len(datasets) for param in exploration_hparams],[])
    num_experiments = [1] * (len(datasets)*len(algos))
    logging_frequency = [min(10, args.T // 5)] * (len(datasets)*len(algos))
    # TODO Ray is not working.
    ray = [False] * (len(datasets)*len(algos))
    algo_arr = sum([[algo] * len(datasets) for algo in algos],[])
    datasets_arr=datasets*len(algos)

    return [
        datasets_arr, training_modes, nn_param_list, linear_model_hparams,
        exploration_hparams, logging_frequency, num_experiments, ray, algo_arr
    ]

args = get_parallel_args(METHODS, DATASETS)
for i in range(NUM_EXPERIMENTS):
    copy_and_run_with_config(
        run_and_plot,
        args,
        WORKING_DIRECTORY,
        parallel=PARALLEL,
        additional_params={
            'reservation':'engs-oxfair20220518',
        },
        # array_parallelism=20,
        array_parallelism=32,
        job_name=JOB_NAME+str(i),
        # time="24:00:00",
        # time="2:00:00",
        time="12:00:00",
        comment="Neurips Deadline",
        partition=PARTITION,
        gres=f"gpu:{GPUS}",
#         constraint='gpu_mem:16GB',
        # mem="100GB",
        # qos="gpu",
        account='engs-oxfair',
    )
