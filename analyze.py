import pickle
import os
import matplotlib.pyplot as plt
import argparse


# from pytorch_experiments import ExperimentResults
from experiments import DATASETS, ExperimentResults

DATASET="Adult"
ALGORITHM="AdversarialPLOT"
ALGOS = ["AdOpt","AdversarialPLOT","PLOT","NeuralUCB","Greedy","Eps_Greedy"]
DATASETS = ["Adult","Bank","MNIST","German","Crime"]

PATH=os.path.join('experiment_results',DATASET,ALGORITHM,'data')

def process_results(data, label):
    # for exp in data:
    #     print(f"{label} FPR")
    #     print([y.fpr.item() for y in exp])
    #     print(f"{label} FnR")
    #     print([y.fnr.item() for y in exp])
    #     print(f"{label} Weight Norm")
    #     print([y.weight_norm for y in exp])
    # for exp in data:
    #     print(f"{label} FPR")
    #     print([y[0].cpu().item() for y in exp])
    #     print(f"{label} FnR")
    #     print([y[1].cpu().item() for y in exp])
    # print(data)
    pass


def process_results_plot(d):
    for exp in d:
        return (
            [y.fpr.item() for y in exp], [y.fnr.item() for y in exp], [y.weight_norm for y in exp]
        )

def process_results_plot_2(d):
    for exp in d:
        return ([y[0].cpu().item() for y in exp], [y[1].cpu().item() for y in exp])


def process_fnr():
    with open(os.path.join(path,"fnr_dump.p"), 'rb') as f:
        x = pickle.load(f)

    #print("FPR + Norm")
    #print(x[-1])

    # train = x[0]
    # process_results(train, "Train")
    # test = x[1]
    # process_results(test, "Test")
    pseudo = x[2]
    import json
    with open(os.path.join(path,"tmp.json"), "w") as f:
        json.dump(pseudo, f)
    # process_results(pseudo, "Pseudo")

# TODO doesn't work exactly?????
def plot_fnr(data_path):
    # with open("fnr_dump.p", 'rb') as f:
    with open(data_path, "rb") as f:
        res = pickle.load(f)

    #print("FPR + Norm")
    #print(x[-1])

    train = res[0]
    train_fpr, train_fnr, _ = process_results_plot(train)

    # test = res[1]
    # test_fpr, test_fnr, _ = process_results_plot(test)
    # plot_helper(x, test_fpr, "test_FPR")

    pseudo = res[2]
    # pseudo_fpr, pseudo_fnr, _ = process_results_plot(pseudo)
    pseudo_fpr, pseudo_fnr = process_results_plot_2(pseudo)

    eps = res[3]
    eps_fpr, eps_fnr, _ = process_results_plot(eps)
    x = [x*10 for x in range(len(pseudo_fpr))]

    print("X len")
    print(len(x))
    print("Pseudo fpr len")
    print(pseudo_fpr)
    print("Eps fpr len")
    print(eps_fpr)
    plot_helper(
        x,
        {
            "Pseudo P(accept|positive)": pseudo_fpr,
            # "Eps FPR": eps_fpr,
        },
        "P(accept|positive)"
    )
    plot_helper(
        x,
        {
            "Pseudo P(accept|negative)": pseudo_fnr,
            # "Eps FNR": eps_fnr,
        },
        "P(accept|negative)"
    )


def plot_helper(x, ys, title):
    plt.figure()
    for label, y in ys.items():
        plt.scatter(x, y, label=label, alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.margins(y=0.1)
    plt.title(f"{title} vs. Time")
    plt.legend()
    plt.savefig(f"{title}.png")


def process_data():
    ''
    with open(os.path.join(PATH,"data_dump.p"), 'rb') as f:
        x = pickle.load(f)
    print("Cumulative Regret: Mean")
    print([round(a, 3) for a in list(x[2].mean_train_cum_regret_averages)])
    print("Cumulative Regret: Std Dev")
    print([round(a, 3) for a in list(x[2].std_train_cum_regret_averages)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET, choices=DATASETS, help="Individual dataset name")
    parser.add_argument('--algorithm', default=ALGORITHM, choices=ALGOS, help="Algorithm name")
    parser_args = parser.parse_args()
    if parser_args.algorithm=="AdOpt":
        parser_args.algorithm="AdversarialPLOT"
    
    PATH=os.path.join('experiment_results',parser_args.dataset,parser_args.algorithm,'data')
    
    process_data()

# process_data()
# process_fnr()
# plot_fnr()
