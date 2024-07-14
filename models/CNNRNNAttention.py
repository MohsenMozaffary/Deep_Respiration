import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Residual block for CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# CNN-RNN-Attention model
class CNNRNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CNNRNNWithAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(32, 32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(64, 64),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.rnn = nn.LSTM(128 * (input_dim // 8) * (input_dim // 8), hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size, 1, seq_len, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.rnn(r_in)
        
        # Attention
        attn_output, _ = self.multihead_attn(r_out, r_out, r_out)
        
        # Aggregating over the sequence length
        out = self.fc(attn_output.mean(dim=1))
        return out