import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class TransE(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(TransE, self).__init__()
        self.name = 'TransE'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm

        self.entities = (nn.Embedding(self.n_ent, self.dims)).to(device)
        self.relations = (nn.Embedding(self.n_rel, self.dims)).to(device)

        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.relations.weight.data,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))

        self.relations.weight.data = nn.functional.normalize(self.relations.weight.data, dim = 1)

    def distance(self, h, r, t):
        data = - torch.norm(h+r-t, dim = 1, p = self.norm)
        return data


    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        
        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])

        pred = self.relations(data[:, 2])

        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        return self.distance(nn.functional.normalize(head, dim = 1), pred, nn.functional.normalize(tail, dim = 1)), self.distance(nn.functional.normalize(cHead, dim = 1), pred, nn.functional.normalize(cTail, dim = 1)), t
