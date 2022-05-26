# Adversarial Optimism for the Bank Loan Problem

## Requirements
Experiments can be either run sequentially by running the experiments.py script or in parallel on a cluster with multiple GPU's by modifying the slurm.py script to suit the particular cluster requirements.

### Sequential Experiments
The experiments.py script runs experiments sequentially and plots the results at the end of the run.
The script supports 5 algorithms: Epsilon Greedy, Greedy, NeuralUCB, PLOT and AdOpt.
The script supports 3 datasets: Adult, Bank, MNIST.
Results are saved in BASE_DIR (currently set to be 'experiment_results')

The algorithms, datasets, number of experiments and number of timesteps can be changed via
ALGOS = ["AdversarialPLOT","PLOT","NeuralUCB","Greedy","Epsilon_Greedy"]
DATASETS = ["Bank","MNIST","Adult"]
TIMESTEPS=2000
NUM_EXPERIMENTS=5
BASE_DIR='experiment_results'

command line - not yet implemented:
To run only one replicate, simply run:
```experiment
python experiments.py --num_experiments=1
```

To run small test experiments (e.g. for dev), one can specify a specific dataset(s), a specific algorithm(s), and a short timescale.
```experiment
python experiments.py --num_experiments=1 --datasets Adult Bank --algo_names AdOpt Greedy --T=2000
```

#### Parallel Experiments
To run experiments in parallel on a cluster use the slurm.py to automatically send SLURM commands to your compute cluster. The settings for the specific experiment are set in the top of the file.

Some settings might need to be adapted to the specific setup.

Run aggregate_and_plot.py on the cluster to aggregate the data from the parallel experiments and create a plot comparing the cummulative regrets from the different algorithms. The settings from slurm.py need to be copied into aggregate_and_plot.py so that it will be able to find the files on the server.

In the paper, we run all algorithms with 5 replicates, for 2000 timesteps. i.e set NUM_EXPERIMENTS=5 TIMESTEPS=2000 in the above.

##### Results
In the paper, we run all algorithms with 5 replicates, for 2000 timesteps.
Hyperparameters are already specified in the model file, and can easily be viewed in `experiments.py`.
In our experiments the algorithm achieved the following performance (regrets are relative to baseline model accuracy):

| Dataset          | Cumulative Regret@T=2000 | Std. Dev of Cumulative Regret@T=2000 |
| ---------------- |------------------------- | -----------------------------------  |
| Adult            |            3.236         |      1.220                           |
| Bank             |            -0.480        |      0.852                           |
| MNIST            |            0.894         |      0.629                           |


The results are output to the BASE_DIR=`experiment_results` folder after running the experiment code.
To print the cumulative regret (mean+stddev) of the method run `analyze.py` script after setting DATASET and ALGORITHM parameters or from command line e.g. `python analyze.py --dataset Adult --algorithm AdOpt`
The plots for individual algorithms are saved in the `figs` directory for each dataset separately, and in addition a multiplot is produced to compare performance of different algorithms.
We also produce two .csv files that record the performance metrics of the biased and adversarially trained model across timesteps, to show the comparatively higher recall of the adversarially trained classifier.
