import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import IPython
import torch
import pickle
from torchvision import datasets, transforms


# @title Data utilities
class DataSet:
    def __init__(self, dataset, labels, num_classes=2, fit_intercept = False):
        self.num_datapoints = dataset.shape[0]
        self.random_state = 0
        self.dataset = dataset
        self.labels = labels
        self.num_classes = num_classes
        self.fit_intercept = fit_intercept
        self.dimension = dataset.shape[1] +fit_intercept


    def get_batch(self, batch_size):
        if batch_size > self.num_datapoints:
            X = self.dataset.values
            Y = self.labels.values
        else:
            X = self.dataset.sample(batch_size, random_state=self.random_state).values
            Y = self.labels.sample(batch_size, random_state=self.random_state).values
        # Y_one_hot = np.zeros((Y.shape[0], self.num_classes))
        # for i in range(self.num_classes):
        #   Y_one_hot[:, i] = (Y == i)*1.0
        self.random_state += 1

        if self.fit_intercept:
            intercept = np.ones(X.shape[0])
            #IPython.embed()
            X = np.concatenate((X, intercept.reshape(len(intercept), 1)), axis = 1)


        if torch.cuda.is_available():
            return (torch.from_numpy(X).to('cuda'), torch.from_numpy(Y).to('cuda'))

        else:
            return (torch.from_numpy(X).float(), torch.from_numpy(Y))

class AdultDataSet:
    def __init__(self, dataset, labels, num_classes=2, fit_intercept = False):
        self.num_datapoints = dataset.shape[0]
        self.random_state = 0
        self.dataset = dataset
        self.labels = labels
        self.num_classes = num_classes
        self.fit_intercept = fit_intercept
        self.dimension = dataset.shape[1] +fit_intercept


    def get_batch(self, batch_size):
        if batch_size > self.num_datapoints:
            X = self.dataset
            Y = self.labels
        else:
            indices=np.random.choice(self.dataset.shape[0], size=batch_size,replace=False)
            X=self.dataset[indices, :]
            Y = self.labels[indices]


        # Y_one_hot = np.zeros((Y.shape[0], self.num_classes))
        # for i in range(self.num_classes):
        #   Y_one_hot[:, i] = (Y == i)*1.0
        # self.random_state += 1
        #
        # if self.fit_intercept:
        #     intercept = np.ones(X.shape[0])
        #     #IPython.embed()
        #     X = np.concatenate((X, intercept.reshape(len(intercept), 1)), axis = 1)
        #

        if torch.cuda.is_available():
            return (torch.from_numpy(X).to('cuda'), torch.from_numpy(Y).to('cuda'))

        else:
            return (torch.from_numpy(X).float(), torch.from_numpy(Y))

class GrowingNumpyDataSet:
    def __init__(self):
        self.dataset_X = None
        self.dataset_Y = None
        self.last_data_addition = None
        self.random_state = 0
        self.dimension = None

    def get_size(self):
        if self.dataset_Y is None:
            return 0
        return len(self.dataset_Y)

    def add_data(self, X, Y):
        if self.dataset_X is None and self.dataset_Y is None:
            self.dataset_X = X
            self.dataset_Y = Y
            self.dimension = X.shape[1]
        else:
            self.dataset_X = torch.cat((self.dataset_X, X), dim=0)
            self.dataset_Y = torch.cat((self.dataset_Y, Y), dim=0)
        # print("shapes")
        # print(self.dataset_X.shape)
        # print(X.shape)
        # print("datasets")
        # print(self.dataset_X)
        # print(X)
        # print(self.dataset_X.shape)

        self.last_data_addition = X.shape[0]

    def pop_last_data(self):
        if self.dataset_X.shape[0] == self.last_data_addition:
            self.dataset_X = None
            self.dataset_Y = None

        else:
            # self.dataset_X = self.dataset_X[: -self.last_data_addition, :]
            # self.dataset_Y = self.dataset_Y[: -self.last_data_addition, :]
            self.dataset_X = self.dataset_X[: -self.last_data_addition]
            self.dataset_Y = self.dataset_Y[: -self.last_data_addition]

    def get_batch(self, batch_size):
        if self.dataset_X is None:
            X = torch.empty(0)
            Y = torch.empty(0)
        elif batch_size > self.dataset_X.shape[0]:
            X = self.dataset_X
            Y = self.dataset_Y
        else:
            indices = random.sample(range(self.dataset_X.shape[0]), batch_size)
            # indices = torch.tensor(indices)
            X = self.dataset_X[indices]
            Y = self.dataset_Y[indices]
        self.random_state += 1
        return (X, Y)


class MNISTDataset:
    def __init__(self, train, batch_size, symbol):

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.symbol = symbol
        self.batch_size = batch_size
        self.dataset = datasets.MNIST(
            "./", train=train, download=False, transform=transform
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

        ### Figure dimension
        (X,Y) = self.get_batch(self.batch_size)
        self.dimension = X.shape[1]


    def get_batch(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError(
                "Provided batch size does not agree with the stored batch size. MNIST."
            )
        [X, Y] = next(iter(self.data_loader))
        Y = (Y == self.symbol) * 1.0
        X = X.view(self.batch_size, -1)
        if torch.cuda.is_available():
            # print("Getting gpu")
            X = X.to('cuda')
            Y = Y.to('cuda')
        return (X, Y)
        # return (X.numpy(), Y.numpy())


class MixtureGaussianDataset:
    def __init__(
        self,
        means,
        variances,
        probabilities,
        theta_stars,
        num_classes=2,
        max_batch_size=10000,
        kernel=lambda a, b: np.dot(a, b),
    ):
        self.means = means
        self.variances = variances
        self.probabilities = probabilities
        self.num_classes = num_classes
        self.theta_stars = theta_stars
        self.cummulative_probabilities = np.zeros(len(probabilities))
        cum_prob = 0
        for i, prob in enumerate(self.probabilities):
            cum_prob += prob
            self.cummulative_probabilities[i] = cum_prob
        self.dimension = theta_stars[0].shape[0]
        self.max_batch_size = max_batch_size
        self.kernel = kernel

    def get_batch(self, batch_size):
        batch_size = min(batch_size, self.max_batch_size)
        X = []
        Y = []
        for _ in range(batch_size):
            val = np.random.random()
            index = 0
            while index <= len(self.cummulative_probabilities) - 1:
                if val < self.cummulative_probabilities[index]:
                    break
                index += 1

            x = np.random.multivariate_normal(
                self.means[index], np.eye(self.dimension) * self.variances[index]
            )
            logit = self.kernel(x, self.theta_stars[index])
            y_val = 1 / (1 + np.exp(-logit))
            y = (np.random.random() >= y_val) * 1.0
            X.append(x)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return (X, Y)


class SVMDataset:
    def __init__(
        self,
        means,
        variances,
        probabilities,
        class_list_per_center,
        num_classes=2,
        max_batch_size=10000,
    ):
        self.means = means
        self.variances = variances
        self.probabilities = probabilities
        self.num_classes = num_classes
        self.class_list_per_center = class_list_per_center
        self.cummulative_probabilities = np.zeros(len(probabilities))
        cum_prob = 0
        for i, prob in enumerate(self.probabilities):
            cum_prob += prob
            self.cummulative_probabilities[i] = cum_prob
        self.max_batch_size = max_batch_size
        self.num_groups = len(self.means)
        self.dim = self.means[0].shape[0]
        self.dimension = self.means[0].shape[0]

    def get_batch(self, batch_size, verbose=False):
        batch_size = min(batch_size, self.max_batch_size)
        X = []
        Y = []
        indices = []
        for _ in range(batch_size):
            val = np.random.random()
            index = 0
            while index <= len(self.cummulative_probabilities) - 1:
                if val < self.cummulative_probabilities[index]:
                    break
                index += 1

            x = np.random.multivariate_normal(
                self.means[index], np.eye(self.dim) * self.variances[index]
            )
            y = self.class_list_per_center[index]
            X.append(x)
            Y.append(y)
            indices.append(index)
        X = np.array(X)
        Y = np.array(Y)
        indices = np.array(indices)

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y)
        if torch.cuda.is_available():
            X = X.to('cuda')
            Y = Y.to('cuda')
        if verbose:
            return (X, Y, indices)
        else:
            return (X, Y)

    def plot(self, batch_size, model=None, names=[], binary = False, title = ""):
        if names == []:
            names = ["" for _ in range(self.num_groups)]
        if self.dim != 2:
            print("Unable to plot the dataset")
        else:
            colors = [
                "blue",
                "red",
                "green",
                "yellow",
                "black",
                "orange",
                "purple",
                "violet",
                "gray",
            ]

            binary_colors = ["black", "red"]

            (X, Y, indices) = self.get_batch(batch_size, verbose=True)
            # print("xvals ", X, "yvals ", Y)
            min_x = float("inf")
            max_x = -float("inf")


            if binary:
                positive_X_filtered_0 = []
                positive_X_filtered_1 = []

                negative_X_filtered_0 = []
                negative_X_filtered_1 = []

                for i in range(self.num_groups):
                    for j in range(len(X)):
                        group_index = indices[j]
                        datapoint_class = self.class_list_per_center[group_index]
                        if datapoint_class:
                            positive_X_filtered_0.append(X[j][0])
                            positive_X_filtered_1.append(X[j][1])
                        else:
                            negative_X_filtered_0.append(X[j][0])
                            negative_X_filtered_1.append(X[j][1])

                        if X[j][0] < min_x:
                            min_x = X[j][0]
                        if X[j][0] > max_x:
                            max_x = X[j][0]

                plt.plot(
                        positive_X_filtered_0,
                        positive_X_filtered_1,
                        "o",
                        color=binary_colors[0],
                        label="Positives",
                    )


                plt.plot(
                        negative_X_filtered_0,
                        negative_X_filtered_1,
                        "o",
                        color=binary_colors[1],
                        label="Negatives",
                    )



            else:
                for i in range(self.num_groups):
                    X_filtered_0 = []
                    X_filtered_1 = []
                    for j in range(len(X)):
                        if indices[j] == i:
                            X_filtered_0.append(X[j][0])
                            if X[j][0] < min_x:
                                min_x = X[j][0]
                            if X[j][0] > max_x:
                                max_x = X[j][0]
                            X_filtered_1.append(X[j][1])

                    plt.plot(
                        X_filtered_0,
                        X_filtered_1,
                        "o",
                        color=colors[i],
                        label="{} {}".format(self.class_list_per_center[i], names[i]),
                    )
            if model is not None:
                # Plot line
                model.plot(min_x, max_x)
            plt.grid(True)
            plt.legend(loc="lower right")
            # IPython.embed()
            plt.title(title)

            plt.show()


def get_batches(protected_datasets, global_dataset, batch_size):
    global_batch = global_dataset.get_batch(batch_size)

    protected_batches = [
        protected_dataset.get_batch(batch_size)
        for protected_dataset in protected_datasets
    ]
    return global_batch, protected_batches


def get_dataset(dataset, batch_size, test_batch_size, fit_intercept):

    if dataset == "Mixture":
        PROTECTED_GROUPS = ["A", "B", "C", "D"]
        d = 20
        means = [
            -10 * np.arange(d) / np.linalg.norm(np.ones(d)),
            np.zeros(d),
            10 * np.arange(d) / np.linalg.norm(np.arange(d)),
            np.ones(d) / np.linalg.norm(np.ones(d)),
        ]
        variances = [0.4, 0.41, 0.41, 0.41]
        theta_stars = [np.zeros(d), np.zeros(d), np.zeros(d), np.zeros(d)]
        probabilities = [0.3, 0.1, 0.5, 0.1]
        kernel = lambda a, b: 0.1 * np.dot(a - b, a - b) - 1

        protected_datasets_train = [
            MixtureGaussianDataset(
                [means[i]], [variances[i]], [1], [theta_stars[i]], kernel=kernel
            )
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            MixtureGaussianDataset(
                [means[i]], [variances[i]], [1], [theta_stars[i]], kernel=kernel
            )
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = MixtureGaussianDataset(
            means, variances, probabilities, theta_stars, kernel=kernel
        )
        test_dataset = MixtureGaussianDataset(
            means, variances, probabilities, theta_stars, kernel=kernel
        )
    elif dataset=="Adult_Continuous":
        x_test = np.load(f'./datasets/datasets_processed/{dataset}_data_test.npy')
        x_train = np.load(f'./datasets/datasets_processed/{dataset}_data_train.npy')
        y_test = np.load(f'./datasets/datasets_processed/{dataset}_data_test_labels.npy')
        y_train = np.load(f'./datasets/datasets_processed/{dataset}_data_train_labels.npy')

        train_dataset = AdultDataSet(x_train, y_train, fit_intercept = fit_intercept)
        test_dataset = AdultDataSet(x_test, y_test, fit_intercept = fit_intercept)
        protected_datasets_train=None
        protected_datasets_test=None

    elif dataset in ["Adult", "German", "Bank", "Crime"]:

        protected_datasets_train = pickle.load(open("./datasets/datasets_processed/{}_protected_train.p".format(dataset), "rb"))
        protected_datasets_test = pickle.load(open("./datasets/datasets_processed/{}_protected_test.p".format(dataset), "rb"))
        train_dataset = pickle.load(  open("./datasets/datasets_processed/{}_train.p".format(dataset), "rb"))
        test_dataset = pickle.load( open("./datasets/datasets_processed/{}_test.p".format(dataset), "rb"))


        train_dataset = DataSet(train_dataset[0], train_dataset[1], fit_intercept = fit_intercept)
        test_dataset = DataSet(test_dataset[0], test_dataset[1], fit_intercept = fit_intercept)

        protected_datasets_train = [DataSet(d[0], d[1], fit_intercept = fit_intercept) for d in protected_datasets_train]
        protected_datasets_test = [DataSet(d[0], d[1], fit_intercept = fit_intercept) for d in protected_datasets_test]


    elif dataset == "MNIST":
        PROTECTED_GROUPS = ["None"]
        protected_datasets_train = [
            MNISTDataset(train=True, batch_size=batch_size, symbol=5)
        ]
        train_dataset = MNISTDataset(train=True, batch_size=batch_size, symbol=5)

        protected_datasets_test = [
            MNISTDataset(train=False, batch_size=test_batch_size, symbol=5)
        ]
        test_dataset = MNISTDataset(train=False, batch_size=test_batch_size, symbol=5)

    elif dataset == "MultiSVM":
        PROTECTED_GROUPS = ["A", "B", "C", "D"]
        d = 2
        means = [
            np.array([0, 5]),
            np.array([0, 0]),
            np.array([5, -2]),
            np.array([5, 5]),
        ]
        variances = [0.5, 0.5, 0.5, 0.5]
        probabilities = [0.3, 0.3, 0.2, 0.2]
        class_list_per_center = [1, 0, 1, 0]

        protected_datasets_train = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
        test_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )

    elif dataset == "SVM":
        PROTECTED_GROUPS = ["A", "B"]
        d = 2
        means = [
            -np.arange(d) / np.linalg.norm(np.arange(d)),
            np.ones(d) / np.linalg.norm(np.ones(d)),
        ]
        variances = [1, 0.1]
        probabilities = [0.5, 0.5]
        class_list_per_center = [0, 1]

        protected_datasets_train = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
        test_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
    else:
        raise ValueError("Unrecognized dataset")

    return (
        protected_datasets_train,
        protected_datasets_test,
        train_dataset,
        test_dataset,
    )


def get_dataset_simple(dataset, batch_size, test_batch_size, fit_intercept ):

    (
        protected_datasets_train,
        protected_datasets_test,
        train_dataset,
        test_dataset,
    ) = get_dataset(dataset, batch_size, test_batch_size, fit_intercept)


    return train_dataset, test_dataset
