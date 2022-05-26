import numpy as np
import matplotlib.pyplot as plt

LINEWIDTH = 3.5
LINESTYLE = "dashed"
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
        linestyle=LINESTYLE,
        linewidth=LINEWIDTH,
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

    else:
        raise ValueError(
            "Decision type not recognized {}".format(exploration_hparams.decision_type)
        )
    return plot_name
