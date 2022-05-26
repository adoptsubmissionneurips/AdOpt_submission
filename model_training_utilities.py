from models import (
    TorchBinaryLogisticRegression,
    get_accuracies_simple,
)
import torch

#@title Useful model training utilities
#Useful utilities. The following function allows to train a model
### for a number of steps.
def gradient_step(model, optimizer, batch_X, batch_y):

    optimizer.zero_grad()
    loss = model.get_loss(batch_X, batch_y)
    loss.backward()
    optimizer.step()

    return model, optimizer

def train_model(
    model,
    num_steps,
    train_dataset,
    batch_size,
    verbose=False,
    restart_model_full_minimization=False,
    weight_decay=0.0
):
    if restart_model_full_minimization:
        model.reinitialize_model()

    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01, weight_decay=weight_decay )

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        model, optimizer = gradient_step(model, optimizer, batch_X, batch_y)

    return model

def evaluate_model(test_dataset, model, threshold):
    with torch.no_grad():
        batch_test = test_dataset.get_batch(10000000000)

        test_accuracy = get_accuracies_simple(
            batch_test,
            model,
            threshold,
        )
    print("Final test model accuracy {}".format(test_accuracy))
    return test_accuracy
