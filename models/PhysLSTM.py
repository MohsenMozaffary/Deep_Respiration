import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysLSTM(nn.Module):
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

            nn.AdaptiveAvgPool3d((None, 1, 1))
            )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=5, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(64, 1)


    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[2]
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.squeeze(x, (3,4))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(batch_size, seq_len)
        return x
