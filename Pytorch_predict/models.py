import torch.nn as nn
import torch
from torch.autograd import Variable
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()      
        self.nn_layers = nn.Sequential(
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
        )
    def forward(self, x):
        # input:   torch.Size([32, 223])
        # output:  torch.Size([32, 20])
        x = self.nn_layers(x)
        return x

class AttModel(nn.Module):
    def __init__(self):
        super(AttModel, self).__init__()
        hidden_dim = 223
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, 1)
        self.nn_layers = nn.Sequential(
            nn.Linear(in_features=223, out_features=4096),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=20),
        )
    def forward(self, x):
        # input:   torch.Size([32, 223])
        # output:  torch.Size([32, 20])

        features = x.unsqueeze(0)
        # Attention:
        batch_size, time_step, hidden_dim = features.size()
        weight = nn.Tanh()(self.dense(features)).squeeze(-1)
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)
        weight = nn.Softmax(dim=1)(weight)
        weight = weight.unsqueeze(1)
        weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        x = features_attention.squeeze(0)
        ## normal dnn
        x = self.nn_layers(x)
        return x

class ResModel(nn.Module): #https://iter01.com/525623.html
    def __init__(self):
        super(ResModel, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.block1  = nn.Sequential(
            nn.Linear(in_features=223, out_features=223),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=223, out_features=223),
            # nn.BatchNorm1d(223)
        )
        self.block2  = nn.Sequential(
            nn.Linear(in_features=20, out_features=20),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=20, out_features=20),
            # nn.BatchNorm1d(20)
        )
        self.nn_layers0  = nn.Sequential(
            nn.Linear(in_features=223, out_features=1024),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=4096),
            # nn.BatchNorm1d(4096)
        )
        self.res1 = nn.Sequential(
            nn.Linear(in_features=223, out_features=4096),
        )
        self.nn_layers = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=20),
        )
        self.res2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=20),
        )
    def forward(self, x):
        # encode
        x = x + self.block1(x)
        x = self.relu(x)
        x = self.nn_layers0(x) + self.res1(x)
        # decode
        x = self.nn_layers(x) + self.res2(x)
        x = x + self.block2(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.nn_gru = nn.Sequential(
            nn.GRU(512, 512, num_layers=2),
        )
        self.linear1 = nn.Linear(223,512)
        self.linear2 = nn.Linear(512,20)
        self.batch_n = nn.BatchNorm1d(512)
        self.activate = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(20)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,dim_feedforward=2048,dropout=0.1)
        encoder_norm = nn.LayerNorm(512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,norm = encoder_norm)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model = 512,nhead=8,dim_feedforward=2048,dropout=0.1)
        decoder_norm = nn.LayerNorm(512)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers=6,norm = decoder_norm)
        # self.self_attention = nn.MultiheadAttention(512, 8)
        # self.layer_norm = nn.LayerNorm(512)
        # self.droput = nn.Dropout(p=0.1)
    def forward(self, x):
        batch_size = x.shape[0]
        src = x.unsqueeze(0)
        src = self.linear1(src)
        # src = self.batch_n(src)
        src = self.activate(src)
        org = src
        x = self.transformer_encoder(src)
        x = self.decoder (org,x)
        x = self.linear2(x)
        x = self.activate(x)
        x = x.view(batch_size, 20)
        return x
    # def forward(self, x):
    #     batch_size = x.shape[0]
    #     src = x.unsqueeze(0)
    #     src = self.linear1(src)
    #     src = batch_n(src)
    #     src = self.activate(src)
    #     # print(src.shape)
    #     out = self.transformer_encoder(src)
    #     # print(out.shape) 
    #     x,_ = self.nn_gru(out)
    #     x = self.linear2(x)
    #     x = self.activate(x)
    #     x = x.view(batch_size, 20)
    #     return x



# Use for data Augment
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=243, out_features=256),           
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(in_features=256, out_features=512),    
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),   
        )
        self.enc_out_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),    
            nn.LeakyReLU(True),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),    
            nn.LeakyReLU(True),
        )
        self.decoder = nn.Sequential(
			nn.Linear(in_features=1024, out_features=512),   
            nn.BatchNorm1d(512), 
            nn.LeakyReLU(True),
			nn.Linear(in_features=512, out_features=256),   
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(in_features=256, out_features=243),
            nn.BatchNorm1d(243),
            nn.LeakyReLU(True),
            nn.Linear(in_features=243, out_features=243)
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar,device):
        std = logvar.mul(0.5).exp_()
        if device=='cpu':
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x,device):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar,device)
        return self.decode(z)



"""
class Attention1(nn.Module):
    # https://github.com/renjunxiang/Multihead-Attention/blob/master/Attention.py
    # 1.输入 [N,T,C] -> Linear、Tanh
    # 2. -> [N,T,1] -> unsqueeze
    # 3. -> [N,T] -> Softmax
    # 4. -> [N,T] -> unsqueeze
    # 5. -> [N,1,T] -> repeat
    # 6. -> [N,C,T] -> transpose
    # 7. -> [N,T,C]
    def __init__(self, hidden_dim):
        super(Attention1, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        batch_size, time_step, hidden_dim = features.size()
        weight = nn.Tanh()(self.dense(features)).squeeze(-1)

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)

        weight = nn.Softmax(dim=1)(weight)
        weight = weight.unsqueeze(1)
        weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention
"""
