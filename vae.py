import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mean, logvar):
        kl_loss = (-0.5 * (logvar - torch.exp(logvar) - torch.pow(mean, 2) + 1)).mean()
        mse_loss = F.mse_loss(x, y)
        return mse_loss + kl_loss


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.mean_linear = nn.Linear(64*7*7, latent_dim)
        self.var_linear = nn.Linear(64*7*7, latent_dim)

    def forward(self, x):
        # Encoding
        x = self.in_conv(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        mean = self.mean_linear(x)
        logvar = self.var_linear(x)
        return mean, logvar


class VAESampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, logvar):
        noise = torch.randn_like(mean, device=mean.device)
        std = torch.exp(0.5 * logvar)
        z = mean + std * noise
        return z


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.projection = nn.Linear(latent_dim, 64*7*7)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(16, 1, 3, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, z):
        # Decoding
        x = self.projection(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.out_conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        """
        :param latent_dim: Latent dimension
        """
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.sampler = VAESampler()
        self.decoder = VAEDecoder(latent_dim)

    def forward(self, x):
        """
        :param x: Ground truth image
        :return: Tuple of (generated image, mean, logvar)
        """
        mean, logvar = self.encoder(x)
        z = self.sampler(mean, logvar)
        y = self.decoder(z)
        return y, mean, logvar
