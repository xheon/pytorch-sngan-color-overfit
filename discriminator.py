import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(3, 16, 3, 2, 1, bias=False)), # 32
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(16),

            torch.nn.utils.spectral_norm(nn.Conv2d(16, 32, 3, 2, 1, bias=False)), # 16
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(32),

            torch.nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 2, 1, bias=False)), # 8
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(64),

            torch.nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1, bias=False)), # 4
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(128),

            torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)),  # 2
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(256),

            torch.nn.utils.spectral_norm(nn.Conv2d(256, 512, 2, 1, 0, bias=False)),  # 1
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(512),

            torch.nn.utils.spectral_norm(nn.Conv2d(512, 1, 1, 1, 0))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)

        return output
