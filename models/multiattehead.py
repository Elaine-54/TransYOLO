import torch.nn as nn
import torch.nn.functional as F
class EFAdecoder(nn.Module):
    def __init__(self,d_model,nhead,dropout,dim_feedforward = 256):
        super(EFAdecoder,self).__init__()
        self.atten1 = nn.MultiheadAttention(d_model,nhead,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
    def with_pos_embed(self, tensor, pos=True):
        return tensor if pos is None else tensor + pos
    def forward(self,src1):#Q,K,V
        q = src1.flatten(-2).permute(2,0,1)
        k = self.with_pos_embed(q)
        v =  self.with_pos_embed(q)
        atten = self.atten1(q,k,v)[0]
        atten = atten+self.dropout(atten)              
        output = self.norm(atten).view(src1.size())
        return output 

class CFADdecoder(nn.Module):
    def __init__(self,d_model,nhead,dropout,dim_feedforward = 256):
        super(CFADdecoder,self).__init__()
        self.atten1 = nn.MultiheadAttention(d_model,nhead,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward,d_model)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
    def with_pos_embed(self, tensor, pos=True):
        return tensor if pos is None else tensor + pos
    def forward(self,src1,src2):#Q,K,V
        q = self.with_pos_embed(src1)
        k =  self.with_pos_embed(src2)
        v = src2
        atten = self.atten1(q,k,v)[0]
        atten = atten+self.dropout(atten)
        atten = self.norm(atten)
        ffn = self.linear2(self.dropout(self.relu(self.linear1(atten))))
        ffn = ffn + self.dropout(ffn)
        output = self.norm(ffn)
        return output 