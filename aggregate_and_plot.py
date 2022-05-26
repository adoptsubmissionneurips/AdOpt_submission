import pickle
import os
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from experiments import ExperimentResults,ExplorationHparams
from multiplot import *

WORKING_DIRECTORY="/cluster/working_directory"
METHODS=["AdversarialPLOT","PLOT","Greedy","Eps_Greedy","NeuralUCB"]
DATASETS=["MNIST","Bank"]
METHODS_FOR_PLOT=METHODS
DATASETS_FOR_PLOT=DATASETS
TIMESTEPS=2000
PARALLEL = True
NUM_EXPERIMENTS=5


DISPLAY_NAMES={
    "AdversarialPLOT":"AdOpt",
    "AdOpt":"AdOpt",
    "PLOT":"PLOT",
    "NeuralUCB":"NeuralUCB",
    "Greedy":"Greedy",
    "Eps_Greedy":"Epsilon Greedy"
}
VERSION = f"_{TIMESTEPS}t_{'_'.join(METHODS)}_{'_'.join(DATASETS)}"
JOB_PREFIX = "exp_"
PARALLEL_STR = "_parallel" if PARALLEL else ""
JOB_NAME = f"{JOB_PREFIX}{PARALLEL_STR}_{VERSION}"

EXPERIMENT_RESULTS_DIR='experiment_results'
FIG_DIRECTORY='figs'
DATA_FILE='data/data_dump.p'
LINEWIDTH = 1
LINESTYLE = "solid"
COLORS={
        'Greedy':'green',
        'NeuralUCB':'grey',
        'AdversarialPLOT':'red',
        'PLOT':'blue',
        'Eps_Greedy':'purple'
    }

@dataclass
class PlotData:
    mean_train_cum_regret_averages: np.ndarray
    std_train_cum_regret_averages: np.ndarray

def aggregate(data_files):
    label = "AdversarialPLOT"
    data = []
    for data_file in data_files:
        with (open(data_file, "rb")) as openfile:
            data.append(pickle.load(openfile))
    timesteps = data[0][0]
    regrets = np.stack([exp[2].mean_train_cum_regret_averages for exp in data])
    means = np.mean(regrets, axis=0)
    stds = np.std(regrets, axis=0)
    return timesteps, means, stds


if __name__=='__main__':
    for dataset in DATASETS:
        plot_data={}
        timesteps={}
        for method in METHODS_FOR_PLOT:
            data_files=[
                os.path.join(WORKING_DIRECTORY,JOB_NAME+str(i),EXPERIMENT_RESULTS_DIR,dataset,method,DATA_FILE) for i in range(NUM_EXPERIMENTS)
            ]
            data = []
            for file in data_files:
                if os.path.exists(file):
                    with (open(file, "rb")) as openfile:
                        data.append(pickle.load(openfile))
                else:
                    print(f'file {data_file} not found - skipping it.')
            timesteps[method] = data[0][0]
            regrets = np.stack([exp[2].mean_train_cum_regret_averages for exp in data])
            means = np.mean(regrets, axis=0)
            stds = np.std(regrets, axis=0)
            plot_data[method]=PlotData(mean_train_cum_regret_averages=means,std_train_cum_regret_averages=stds)

        plot_name=f'{dataset}_regrets_{"_".join(METHODS_FOR_PLOT)}_{TIMESTEPS}T'
        base_figs_directory=os.path.join(WORKING_DIRECTORY,FIG_DIRECTORY)

        if not os.path.isdir(base_figs_directory):
            try:
                os.makedirs(base_figs_directory)
            except OSError:
                print("Creation of figs directories failed")
            else:
                print("Successfully created the figs directory")
        for label in METHODS_FOR_PLOT:
            plot_results(
                timesteps=timesteps[label],
                experiment_results=plot_data[label],
                network_type='multiple',
                base_figs_directory=base_figs_directory,
                dataset=dataset,
                training_mode='',
                exploration_hparams=ExplorationHparams(),
                color=COLORS[label],
                label=DISPLAY_NAMES[label]
            )
        plt.title(f'{dataset}')
        print(f'saving figure to {os.path.join(base_figs_directory,plot_name+".png")}')
        plt.savefig(
            os.path.join(base_figs_directory,plot_name+'.png'),
            bbox_inches="tight",
            dpi=300
        )
        plt.close('all')

        