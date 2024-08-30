#TuckerNet for respiratory rate estimation
import torch
import torch.nn as nn
import torch.nn.init as init

class TuckerNet(nn.Module):

    def __init__(self, in_channels=1, n_filters1=16, n_filters2=32, dropout_rate1=0.15,
                 dropout_rate2=0.15, pool_size=(2, 2), nb_dense=128):
        """Definition of TuckerNet.
        Args:
          in_channels: number of input video frame channels. Default: 1
        Returns:
          TuckerNet model.
        """
        super(TuckerNet, self).__init__()
        self.in_channels = in_channels
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.nb_dense = nb_dense
        # Spatial conv2ds
        self.conv1 = nn.Conv2d(
                                self.in_channels, 
                                self.n_filters1, 
                                kernel_size=5,
                                padding=(2, 2), bias=True
                              )
        
        self.conv2 = nn.Conv2d(
                                self.n_filters1, 
                                self.n_filters1, 
                                kernel_size=5, 
                                bias=True
                              )
        
        self.conv3 = nn.Conv2d(
                                self.n_filters1, 
                                self.n_filters2, 
                                kernel_size=3, 
                                padding=(1, 1)
                              )
        
        self.conv4 = nn.Conv2d(
                                self.n_filters2, 
                                self.n_filters2, 
                                kernel_size=3, 
                                bias=True
                              )
        
        # Average pooling layers
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1,1))
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Sequential layers
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=14, num_layers=5, dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=30, hidden_size=16, num_layers=3, dropout=0.1, batch_first=True,  bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(32, 1)
        # Initializing weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                elif isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0.0)
    # inputs is a tensor of one video successive frame differences
    # inputs shape: batch, time, channels, width, height
    # t1, and t2 are time loading vectors from Tucker decomposition
    # t1, and t2 shape: time
    def forward(self, inputs, t1, t2):
        diff_input = inputs

        t1 = torch.unsqueeze(t1, -1)
        t2 = torch.unsqueeze(t2, -1)

        d1 = torch.tanh(self.conv1(diff_input))
        d2 = torch.tanh(self.conv2(d1))
        d3 = self.avg_pooling_1(d2)
        d4 = self.dropout_1(d3)

        d5 = torch.tanh(self.conv3(d4))
        d6 = torch.tanh(self.conv4(d5))

        d7 = self.avg_pooling_3(d6)
        d8 = self.dropout_3(d7)
        d9 = self.adaptive_avg_pool2d(d8)
        d9 = torch.squeeze(d9)
        d9 = torch.unsqueeze(d9, -1)
        d9 = d9.permute(2, 0, 1)
        d11, _ = self.lstm1(d9)

        d11 = torch.cat((t1, t2, d11), dim=2)
        d12, _ = self.lstm2(d11)
        d13 = self.fc(d12)
        out = d13.squeeze()

        return out
