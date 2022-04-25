import os, sys
import time, glob
import random
import numpy as np
import torch

sys.path.append('../../JKLib')
from torchJK import CosineAnnealingWarmUpRestarts, PositionalEncoding

class TransformerION(torch.nn.Module):
    def __init__(self,
                 num_encoder_layers: int,                 
                 d_model: int,
                 nhead: int,                                  
                 dim_feedforward: int,
                 n_parameters: int,
                 dropout: float = 0.1):
        super(TransformerION, self).__init__()
        emb_size = d_model
                
        self.conv1 = torch.nn.Conv1d(in_channels=n_parameters, out_channels=256, kernel_size=5, 
                  	stride=3, padding=0, dilation=2, 
                  	groups=1, bias=True, padding_mode='zeros')        
        self.conv1_bn = torch.nn.BatchNorm1d(256)
        self.relu1 = torch.nn.ReLU(inplace=False)
        
        self.conv2 = torch.nn.Conv1d(in_channels=256, out_channels=d_model, kernel_size=5, 
                  	stride=2, padding=0, dilation=2, 
                  	groups=1, bias=True, padding_mode='zeros')
        self.conv2_bn = torch.nn.BatchNorm1d(512)
        self.relu2 = torch.nn.ReLU(inplace=False)
        
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.conv3 = torch.nn.Conv1d(in_channels=d_model, out_channels=256, kernel_size=5, 
                  	stride=2, padding=0, dilation=2, 
                  	groups=1, bias=True, padding_mode='zeros')
        self.conv3_bn = torch.nn.BatchNorm1d(256)
        self.relu3 = torch.nn.ReLU(inplace=False)        
        self.conv4 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, 
                  	stride=2, padding=0, dilation=2, 
                  	groups=1, bias=True, padding_mode='zeros')
        self.conv4_bn = torch.nn.BatchNorm1d(128)
        self.relu4 = torch.nn.ReLU(inplace=False)
        
        self.generator = torch.nn.Linear(512, n_parameters)
        
    def forward(self, src: torch.Tensor):

        out = self.conv1(src)
        out = self.conv1_bn(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu2(out)
        
        out = out.permute(0,2,1)     
        out = self.positional_encoding(out)           
        out = self.transformer_encoder(out)        
        out = out.permute(0,2,1)
        
        out = self.conv3(out)        
        out = self.conv3_bn(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = self.relu4(out)
        
        # print(out.size())
# #         outs = outs.squeeze(2)          
        out = out.flatten(1)        
        out = self.generator(out)

        return out