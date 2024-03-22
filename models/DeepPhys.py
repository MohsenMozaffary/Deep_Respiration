"""DeepPhys for thermal camera
This is implementation of the following paper:
    DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
    ECCV, 2018
    Weixuan Chen, Daniel McDuff
"""

import numpy as np
import torchvision
import torch
from torch import nn as nn
import torch.nn.functional as F

class DeepPhys(nn.Module):
    def __init__(self, image_size = 72, bias = True):
        """DeepPhys parameters
        image_size is the width and height of the image.
        Input video will be batch_size, time, channel, width, and height
        As thermal cameras have only 1 channel, the starting Conv2d is set to 1 channel input
        """

        super(DeepPhys, self).__init__()
        # Motion part
        self.m_conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1, bias = bias)
        self.m_conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1, bias = bias)
        self.m_conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1, bias = bias)
        self.m_conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = bias)
        # Appearance part
        self.a_conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1, bias = bias)
        self.a_conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1, bias = bias)
        self.a_conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1, bias = bias)
        self.a_conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = bias)
        # Attention part
        self.attention1 = nn.Conv2d(32, 1, kernel_size = 1, padding = 0, bias = bias)
        self.attention2 = nn.Conv2d(64, 1, kernel_size = 1, padding = 0, bias = bias)
        # Linear part
        dense_features = int(int(image_size/4) * int(image_size/4) * 64)
        self.dense1 = nn.Linear(dense_features, 128, bias = bias)
        self.dense2 = nn.Linear(128, 1, bias = bias)


    def forward(self, x):
        # Batch_size, time, 1, height, width
        motion = x[0]
        # Batch_size, time, 1, height, width
        appearance = x[1]
        batch_size, T, C, Hm, Wm = motion.shape
        batch_size, T, C, Ha, Wa = appearance.shape
        # Batch_size x time, 1, height, width
        motion = motion.view(batch_size*T, C, Hm, Wm)
        # Batch_size x time, 1, height, width
        appearance = appearance.view(batch_size*T, C, Ha, Wa)
        # Batch_size x time, 1, height, width -> Batch_size x time, 32, height, width
        motion = torch.tanh(self.m_conv1(motion))
        # Batch_size x time, 32, height, width -> Batch_size x time, 32, height, width
        motion = torch.tanh(self.m_conv2(motion))
        # Batch_size x time, 1, height, width -> Batch_size x time, 32, height, width
        appearance = torch.tanh(self.a_conv1(appearance))
        # Batch_size x time, 32, height, width -> Batch_size x time, 32, height, width
        appearance = torch.tanh(self.a_conv2(appearance))
        # Batch_size x time, 32, height, width -> Batch_size x time, 1, height, width
        attention1 = torch.sigmoid(self.attention1(appearance))
        # shape of mean_frames1 = Batch_size x time, 1
        mean_frames1 = attention1.sum(dim=(2,3))/(attention1.shape[2]*attention1.shape[3]*0.5)
        #Batch_size x time, 1, height, width -> Batch_size x time, 1, height, width
        attention1 = attention1/mean_frames1
        #Batch_size x time, 32, height, width -> Batch_size x time, 32, height/2, width/2
        motion = F.dropout(F.avg_pool2d(motion*attention1, kernel_size=2), p = 0.2)
        #Batch_size x time, 32, height, width -> Batch_size x time, 32, height/2, width/2
        appearance = F.dropout(F.avg_pool2d(appearance, kernel_size = 2), p = 0.2)
        #Batch_size x time, 32, height/2, width/2 -> Batch_size x time, 64, height/2, width/2
        motion = torch.tanh(self.m_conv3(motion))
        #Batch_size x time, 64, height/2, width/2 -> Batch_size x time, 64, height/2, width/2
        motion = torch.tanh(self.m_conv4(motion))
        #Batch_size x time, 32, height/2, width/2 -> Batch_size x time, 64, height/2, width/2
        appearance = torch.tanh(self.a_conv3(appearance))
        #Batch_size x time, 64, height/2, width/2 -> Batch_size x time, 64, height/2, width/2
        appearance = torch.tanh(self.a_conv4(appearance))
        #Batch_size x time, 64, height/2, width/2 -> Batch_size x time, 1, height/2, width/2
        attention2 = torch.sigmoid(self.attention2(appearance))
        # shape of mean_frames1 = Batch_size x time, 1
        mean_frames2 = attention2.sum(dim=(2,3))/(attention2.shape[2]*attention2.shape[3]*0.5)
        #Batch_size x time, 1, height/2, width/2 -> Batch_size x time, 1, height/2, width/2
        attention2 = attention2/mean_frames2
        #Batch_size x time, 64, height/2, width/2 -> Batch_size x time, 64, height/4, width/4
        motion = F.dropout(F.avg_pool2d(motion*attention2, kernel_size=2), p=0.2)
        #Batch_size x time, 64, height/4, width/4 -> Batch_size x time, 64 x height/4 x width/4
        motion = motion.view(motion.shape[0], -1)
        #Batch_size x time, 64 x height/4 x width/4 -> Batch_size x time, 128
        motion = F.dropout(torch.tanh(self.dense1(motion)), p=0.2)
        #Batch_size x time, 128 -> Batch_size x time, 1
        motion = torch.tanh(self.dense2(motion))
        #Batch_size x time, 128 -> #Batch_size, time, 1
        motion = motion.view(batch_size, T, -1)
        
        return motion
