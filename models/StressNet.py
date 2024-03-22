import numpy as np
import torchvision
import torch
from torch import nn as nn


'''
StressNet model for respiratory signal extraction from thermal camera.
'''

class StressNet(nn.Module):
    def __init__(self, resnet_layers = 8, lstm_input_size = 256, 
                 lstm_hidden_size = 3, lstm_num_layers = 2, fps = 10, lstm_segment_duration=1):
        super().__init__()
        self.fps = fps # frames per second.
        self.CNN_block = nn.Sequential([
            # Taking the first 8 layers of resnet50
            *list(torchvision.models.resnet50(pretrained=True, progress=True).children())[:resnet_layers],
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        ]) # A pre-trained resnet50 block for early layers of the model.
        # LSTM layer for sequential processing of the resnet50's features.
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_segment_duration*fps)

    def forward(self, x):
        # input data shape: batch_size, time, channels, height, width
        batch_size, T, C, H, W = x.shape
        # batch_size, time, channels, height, width -> batch_size x time, channels, height, width
        x = x.view(batch_size*T, C, H, W)
        x = self.CNN_block(x)
        x = x.view(batch_size, T//self.fps, -1)
        x, _ = self.lstm(x)
        x = x[-1].view(batch_size, T, -1).squeeze()

        return x