import torch.nn as nn
import torch.nn.functional as F
import torch
import math

_SQRT2 = math.sqrt(2)


class ADSH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quan_loss = ((F - B[omega, :]) ** 2).sum()
        loss = (hash_loss + self.gamma * quan_loss) / (F.shape[0] * B.shape[0])

        return loss


class SP_Loss(nn.Module):
    def __init__(self, device):
        super(SP_Loss, self).__init__()
        self.device = device
    
    def forward(self, sp_v):
        # sp_v = F.softmax(sp_v, dim=2)
        sp_loss = torch.tensor(0).to(self.device)
        att_size = sp_v.shape[1]
        cnt = 0
        for i in range(att_size-1):
            for j in range(i+1, att_size):
                sp_loss = sp_loss + torch.norm(torch.sqrt(sp_v[:,i,:]) - torch.sqrt(sp_v[:,j,:]), dim = 1)
                cnt += 1
        sp_loss = sp_loss / _SQRT2 / cnt
        sp_loss = (1 - sp_loss).sum()
        return sp_loss


class CH_Loss(nn.Module):
    def __init__(self, device, t=0.4):
        super(CH_Loss, self).__init__()
        self.t = t
        self.relu = nn.ReLU(inplace=True)
        self.device = device

    def forward(self, ch_v):
        # ch_v = F.softmax(ch_v, dim=2)
        ch_loss = torch.zeros(ch_v.shape[0]).to(self.device)
        att_size = ch_v.shape[1]
        cnt = 0
        for i in range(att_size-1):
            for j in range(i+1, att_size):
                ch_loss = ch_loss + torch.norm(torch.sqrt(ch_v[:,i,:]) - torch.sqrt(ch_v[:,j,:]), dim = 1)
                cnt += 1
        ch_loss = ch_loss / _SQRT2 / cnt
        ch_loss = self.relu(self.t - ch_loss).sum()
        return ch_loss


class Exch_Loss(nn.Module):
    def __init__(self, code_length, device, lambd_sp=0.1, lambd_ch=0.1, lambd_al=1.0, gamma=200, t=0.4):
        super(Exch_Loss, self).__init__()
        self.ch_loss = CH_Loss(device, t)
        self.sp_loss = SP_Loss(device)
        self.lambd_sp = lambd_sp
        self.lambd_ch = lambd_ch
        self.lambd_al = lambd_al
        self.gamma = gamma
        self.code_length = code_length
        self.aligning = False
        self.device = device
        self.quanting = False

    def forward(self, F, B, S, omega, sp_v, ch_v, avg_local_f, batch_anchor_local_f):
        ch_loss = self.ch_loss(ch_v) /  F.shape[0] * self.lambd_ch
        sp_loss = self.sp_loss(sp_v) /  F.shape[0] * self.lambd_sp
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / F.shape[1] * 12
        quan_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / F.shape[1] * 12
        # loss = (hash_loss + self.lambd * sp_loss + self.gamma * ch_loss) / (F.shape[0] * B.shape[0])
        # loss = hash_loss + quan_loss + sp_loss + ch_loss
        loss = hash_loss + sp_loss + ch_loss
        if self.aligning:
            align_loss = torch.norm(avg_local_f-batch_anchor_local_f) / F.shape[0] * self.lambd_al
            loss += align_loss
        else:
            align_loss = torch.tensor([0.0]).to(self.device)
        if self.quanting:
            loss += quan_loss
        return loss, hash_loss, quan_loss, sp_loss, ch_loss, align_loss


