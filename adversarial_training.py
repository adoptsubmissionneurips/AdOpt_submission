import torch
import numpy as np
from performance_metrics import metrics_adversarial
import math
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


# loss with class weights
def BCELoss_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * \
              weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss

def adversarial_training(classifier, generator, discriminator, n_epochs, lambd,
                         x_test, x_train, y_train, device,
                         adv_loss_type, opt_params, verbose):
    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]
    # set weighted loss coefficient
    weight_ratio = torch.tensor([1., 1.])
    adv_weight_ratio = torch.tensor([1., n_samples_train / n_samples_test])
    if verbose:
        print("adversarial weight ratio", adv_weight_ratio)
    # setting weights for adversarial and classifier components of the loss
    class_weight = 1
    adv_weight = lambd

    x_test.to(device)
    x_train.to(device)
    x_test=x_test.to(torch.float32)
    x_train=x_train.to(torch.float32)
    y_train.to(device)
    classifier.to(device)
    generator.to(device)
    discriminator.to(device)
    # create dataloaders
    data_train = TensorDataset(x_train, y_train)
    trainloader = DataLoader(data_train, batch_size=opt_params['batch_size'])

    # Create mixed dataset with labels indicating whether the data came from test
    # (1) or train (0), and create the dataloader
    x_test_labelled = torch.cat(
        (x_test, torch.ones(x_test.shape[0]).unsqueeze(1).to(device)), dim=1).to(torch.float32)
    x_train_labelled = torch.cat(
        (x_train, torch.zeros(x_train.shape[0]).unsqueeze(1).to(device)),
        dim=1).to(torch.float32)
    x = torch.cat((x_test_labelled, x_train_labelled)).to(device)
    # setting placeholder test labels
    y_test= torch.zeros(x_test.shape[0]).to(device)
    y = torch.cat((y_test, y_train)).to(device)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=opt_params['batch_size'],
                        shuffle=True)

    # Establish optimisers for generator, discriminator, and classifier
    gen_optm = torch.optim.Adam(generator.parameters(),
                                lr=opt_params['gen_lr'],
                                betas=opt_params['betas'])
    # Multiplies to learning rate of 5 for disc and 10 for clas were found
    # empirically
    disc_optm = torch.optim.Adam(discriminator.parameters(),
                                 lr=opt_params['disc_lr'],
                                 betas=opt_params['betas'])
    clas_optm = torch.optim.Adam(classifier.parameters(),
                                 lr=opt_params['clas_lr'],
                                 betas=opt_params['betas'])

    # Train the classifier, generator, and discriminator
    for j, epoch in enumerate(range(n_epochs)):

        clas_running_loss = 0.0
        adv_running_loss = 0.0

        for i, batch in enumerate(loader):

            inputs = batch[0][:, :-1]
            labels_y = batch[1].unsqueeze(1)
            labels_z = batch[0][:, -1].unsqueeze(1)
            inputs, labels_y, labels_z = \
                inputs.to(device), labels_y.to(device), labels_z.to(device)

            clas_optm.zero_grad()
            gen_optm.zero_grad()
            disc_optm.zero_grad()

            # Encode the data and classify the encoded data
            encoded = generator(inputs)
            clas_outputs = classifier(encoded)
            disc_output = discriminator(encoded)

            indic = labels_z == torch.zeros(labels_z.shape).to(device)
            inv_indic = indic == torch.zeros(labels_z.shape).to(device)
            clas_outputs_no_test = clas_outputs[indic]
            labels_y_no_test = labels_y[indic]
            weighted_loss = BCELoss_weighted(weight_ratio)
            clas_loss = weighted_loss(clas_outputs_no_test, labels_y_no_test)
            if adv_loss_type == 'BCE':
                # Weighted loss accounts for difference in train/test set sizes
                adv_weighted_loss = BCELoss_weighted(adv_weight_ratio)
                adv_loss = -adv_weighted_loss(disc_output, labels_z)

            elif adv_loss_type == 'WGAN':
                adv_loss = (torch.mean(disc_output[inv_indic]) -
                            torch.mean(disc_output[indic]))
                # gradient penalty - controls the amount of regularisation
                if opt_params['reg'] > 0:
                    grads = torch.autograd.grad(adv_loss,
                                                generator.parameters(),
                                                create_graph=True)
                    grad = torch.cat([g.view(-1) for g in grads]).pow(2).sum()
                    (opt_params['reg'] * grad).backward(retain_graph=True)
                gen_optm.zero_grad()

            loss = class_weight * clas_loss + adv_weight * adv_loss

            # Back propagate
            loss.backward()

            # Multiply discriminator gradients by -1, as discriminator is
            # maximising the loss
            for p_d in discriminator.parameters():
                p_d.grad *= -1

            clas_optm.step()
            gen_optm.step()
            disc_optm.step()

            clas_running_loss += clas_loss.item()
            adv_running_loss += adv_loss.item()

    if verbose:

        # Create dataset with labels indicating whether the data came from test
        # (1) or train (0) for evaluating the discriminator performance
        x_eval = torch.cat((x_test, x_train))
        y_eval = torch.cat(
            (torch.ones(x_test.shape[0]), torch.zeros(x_train.shape[0])))
        eval_dataset = TensorDataset(x_eval, y_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=x_eval.size(dim=0),
                                 shuffle=True)

        if adv_loss_type == 'BCE':
            disc_metrics = metrics_adversarial(x_eval, y_eval, discriminator, generator)
            print("discriminator metrics", disc_metrics)

    return classifier, generator, discriminator
