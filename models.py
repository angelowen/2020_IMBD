import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=223, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64,20)
        self.nn_layers = nn.Sequential(
            # ========================================================== #
            # fully connected layer
            # Can stack number of layers you want
            # Note that the first layer's in_features need to match to data's dim.
            # And out_features need to match to label's dim
            nn.Linear(in_features=223, out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=20),
            # ========================================================== #
        )
# nn.ELU
    def forward(self, x):
        # data fit into model, no need to rewrite
        # input:   torch.Size([32, 223])
        # output:  torch.Size([32, 20])
        x = self.nn_layers(x)

        # print(x.shape)
        return x


