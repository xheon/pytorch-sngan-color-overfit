import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dimension: int) -> None:
        super().__init__()

        self.verbose = True

        self.model = nn.Sequential(
            nn.Conv2d(latent_dimension, 64, 1, 1, 0, bias=False), # 1
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 2, 2, 0, bias=False), # 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), # 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), # 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), # 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), # 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z

        for depth, module in enumerate(self.model.children()):
            shape_before = x.size()
            x = module(x)
            shape_after = x.size()
            if self.verbose is True:
                print(f"{depth:02d}: {shape_before} --> {shape_after}")

        self.verbose = False

        output = x

        # output: torch.Tensor = self.model(z)

        return output
