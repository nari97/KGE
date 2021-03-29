import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class NLLLoss(Loss):

    def __init__(self, adv_temperature=None, margin=6.0):
        super(NLLLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.loss = nn.MarginRankingLoss(margin)
        self.criterion = nn.Softplus()
        

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        p = self.criterion(-p_score)
        n = self.criterion(n_score)

        return (p.mean() + n.mean())/2

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score