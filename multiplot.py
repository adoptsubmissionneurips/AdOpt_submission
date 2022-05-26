import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from experiments import ExperimentResults,ExplorationHparams

STD_GAP = 0.5
ALPHA = 0.1

def plot_helper(timesteps, accuracies, accuracies_stds, label, color, broadcast=False):
    if broadcast:
        accuracies = np.array([accuracies] * len(timesteps))
        accuracies_stds = np.array([accuracies_stds] * len(timesteps))
    plt.plot(
        timesteps,
        accuracies,
        label=label,
        linestyle='solid',
        linewidth=1,
        color=color,
    )
    plt.fill_between(
        timesteps,
        accuracies - STD_GAP * accuracies_stds,
        accuracies + STD_GAP * accuracies_stds,
        color=color,
        alpha=ALPHA,
    )

def plot_title(
    plot_type,
    dataset,
    network_type,
    training_mode,
    exploration_hparams,
):
    if plot_type == "accuracy":
        plot_type_prefix = "Test and Train Accuracies"
        plot_type_file_prefix = "test_train_accuracies"
    elif plot_type == "regret":
        plot_type_prefix = "Regret"
        plot_type_file_prefix = "regret"
    elif plot_type == "loss":
        plot_type_prefix = "Loss"
        plot_type_file_prefix = "loss"

    if exploration_hparams.decision_type == "simple":
        if exploration_hparams.epsilon_greedy:
            plt.title(
                (
                    f"{plot_type_prefix} {dataset} - "
                    f"Epsilon Greedy {exploration_hparams.epsilon} - {network_type} - {training_mode}"
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_epsgreedy_{}_{}_{}".format(
                dataset,
                plot_type_file_prefix,
                exploration_hparams.epsilon,
                network_type,
                training_mode,
            )
        if exploration_hparams.adjust_mahalanobis:
            plt.title(
                (
                    f"{plot_type_prefix} {dataset} - Optimism alpha {exploration_hparams.alpha} "
                    f"- Mreg {exploration_hparams.mahalanobis_regularizer} "
                    f"- Mdisc {exploration_hparams.mahalanobis_discount_factor} - "
                    f"{network_type} - {training_mode}"
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_optimism_alpha_{}_mahreg_{}_mdisc_{}_{}_{}".format(
                dataset,
                plot_type_file_prefix,
                exploration_hparams.alpha,
                exploration_hparams.mahalanobis_regularizer,
                exploration_hparams.mahalanobis_discount_factor,
                network_type,
                training_mode,
            )
        if (
            not exploration_hparams.epsilon_greedy and not exploration_hparams.adjust_mahalanobis
        ):
            plt.title(
                "{} {} - {} - {} ".format(
                    plot_type_prefix, dataset, network_type, training_mode
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_biased_{}_{}".format(
                dataset, plot_type_file_prefix, network_type, training_mode
            )
    elif exploration_hparams.decision_type == "counterfactual":
        plt.title(
            "{} {} - {} - {} - {}".format(
                plot_type_prefix,
                dataset,
                network_type,
                training_mode,
                exploration_hparams.decision_type,
            ),
            fontsize=8,
        )
        plot_name = "{}_{}_biased_{}_{}_{}".format(
            dataset,
            plot_type_file_prefix,
            network_type,
            training_mode,
            exploration_hparams.decision_type,
        )
    elif exploration_hparams.decision_type == "adversarial_counterfactual":
        plt.title(
            "{} {} - {} - {} - {}".format(
                plot_type_prefix,
                dataset,
                network_type,
                training_mode,
                exploration_hparams.decision_type,
            ),
            fontsize=8,
        )
        plot_name = "{}_{}_biased_{}_{}_{}".format(
            dataset,
            plot_type_file_prefix,
            network_type,
            training_mode,
            exploration_hparams.decision_type,
        )

    else:
        raise ValueError(
            "Decision type not recognized {}".format(exploration_hparams.decision_type)
        )
    return plot_name


def plot_results(
    timesteps,
    experiment_results,
    network_type,
    base_figs_directory,
    dataset,
    training_mode,
    exploration_hparams,
    color,
    label,
    plot_legend=True
):
    # ACCURACY PLOTS
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_test_biased_accuracies_cum_averages,
    #     experiment_results.std_test_biased_accuracies_cum_averages,
    #     "Biased Model Test - no decision adjustment",
    #     "blue",
    # )
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_accuracies_cum_averages,
    #     experiment_results.std_accuracies_cum_averages,
    #     label="Unbiased Model Test - all data train",
    #     color="red",
    # )
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_train_biased_accuracies_cum_averages,
    #     experiment_results.std_train_biased_accuracies_cum_averages,
    #     label="Online Biased Model - filtered data train",
    #     color="violet",
    # )
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_accuracy_validation_baseline_summary,
    #     experiment_results.std_accuracy_validation_baseline_summary,
    #     label="Baseline Accuracy",
    #     color="black",
    #     broadcast=True
    # )
    #
    # plot_name = plot_title(
    #     "accuracy",
    #     dataset,
    #     network_type,
    #     training_mode,
    #     exploration_hparams,
    # )
    # plt.xlabel("Timesteps")
    # plt.ylabel("Accuracy")
    # lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
    # print(f"Saving plot to {base_figs_directory}/{plot_name}.png")
    # plt.savefig(
    #     "{}/{}.png".format(base_figs_directory, plot_name),
    #     bbox_extra_artists=(lg,),
    #     bbox_inches="tight",
    # )
    # plt.close("all")

    # REGRET PLOTS
    plot_helper(
        timesteps,
        experiment_results.mean_train_cum_regret_averages,
        experiment_results.std_train_cum_regret_averages,
        label=label,
        color=color,
    )
    plot_name = plot_title(
        "regret",
        dataset,
        network_type,
        training_mode,
        exploration_hparams,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Regret")
    if plot_legend:
        lg = plt.legend(fontsize=8, loc="upper left")
    # plt.savefig(
    #     "{}/{}.png".format(base_figs_directory, plot_name),
    #     bbox_extra_artists=(lg,),
    #     bbox_inches="tight",
    # )
    # plt.close("all")

    # LOSS PLOTS
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_loss_validation_averages,
    #     experiment_results.std_loss_validation_averages,
    #     label="Unbiased model loss",
    #     color="blue",
    # )
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_loss_validation_biased_averages,
    #     experiment_results.std_loss_validation_biased_averages,
    #     label="Biased model loss",
    #     color="red",
    # )
    # plot_helper(
    #     timesteps,
    #     experiment_results.mean_loss_validation_baseline_summary,
    #     experiment_results.std_loss_validation_baseline_summary,
    #     label="Baseline Loss",
    #     color="black",
    #     broadcast=True
    # )
    # plot_name = plot_title(
    #     "loss",
    #     dataset,
    #     network_type,
    #     training_mode,
    #     exploration_hparams,
    # )
    # plt.xlabel("Timesteps")
    # plt.ylabel("Loss")
    # lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")

    # plt.savefig(
    #     "{}/{}.png".format(base_figs_directory, plot_name),
    #     bbox_extra_artists=(lg,),
    #     bbox_inches="tight",
    # )
    # plt.close("all")

if __name__=='__main__':
    LINEWIDTH = 1
    LINESTYLE = "solid"
    STD_GAP = 0.5
    ALPHA = 0.1

    if suffix!='':
        base_results_dir='experiment_results_'+suffix
    else:
        base_results_dir='experiment_results'

    results_dir=os.path.join(base_results_dir,DATASET)
    data_file='data/data_dump.p'

    colors={
        'Greedy':'blue',
        'NeuralUCB':'red',
        'AdversarialPLOT':'orange',
        'PLOT':'cyan',
        'Eps_Greedy':'purple'
    }

    base_figs_directory=os.path.join(results_dir,'figs')

    if not os.path.isdir(base_figs_directory):
        try:
            os.makedirs(base_figs_directory)
        except OSError:
            print("Creation of figs directories failed")
        else:
            print("Successfully created the figs directory")
    plot_name='_'.join([DATASET]+experiments)
    data={}
    for label in experiments:
        data[label]=[]
        filename=os.path.join(results_dir,label,data_file)
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    data[label].append(pickle.load(openfile))
                except EOFError:
                    break
    results={label:data[label][0][2] for label in experiments}
    timesteps={label:data[label][0][0] for label in experiments}
    for label in experiments:
        plot_results(
            timesteps=timesteps[label],
            experiment_results=results[label],
            network_type='multiple',
            base_figs_directory=os.path.join(results_dir,'figs'),
            dataset=DATASET,
            training_mode='',
            exploration_hparams=ExplorationHparams(),
            color=colors[label],
            label=label
        )
    print('saving figure to {}'.format(os.path.join(base_figs_directory,plot_name+'.png')))
    plt.savefig(
        os.path.join(base_figs_directory,plot_name+'.png'),
        bbox_inches="tight",
    )
    plt.show()
