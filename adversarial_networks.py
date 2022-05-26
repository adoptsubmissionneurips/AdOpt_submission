import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, representation_layer_size):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(input_dim, representation_layer_size),
            nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            nn.Linear(representation_layer_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main_module(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, representation_layer_size, loss):
        self.loss = loss
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(input_dim, representation_layer_size),
            nn.ReLU(),
            nn.Linear(representation_layer_size, representation_layer_size),
            nn.ReLU(),
            nn.Linear(representation_layer_size, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.loss == 'BCE':
            return self.sigmoid(self.main_module(x))
        if self.loss == 'WGAN':
            return self.main_module(x)

class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main_module(x)
