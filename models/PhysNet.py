import torch
import torch.nn as nn

'''
PhysNet implementation for respiratory signal estimation through thermal camera.
'''

class PhysNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2), padding = 0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size = (2, 2, 2), stride = 2, padding = 0),

            nn.Conv3d(64, 64, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = 0),
            
            nn.Conv3d(64, 64, kernel_size = (3, 3, 3,), stride = 1, padding = 0),
            nn.BatchNorm3d(64),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = (2, 1, 1), stride = (2, 1, 1), padding = (1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(64, 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)

        return out