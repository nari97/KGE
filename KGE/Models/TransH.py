import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class TransH(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(TransH, self).__init__()
        self.name = 'TransH'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm

        self.entities = (nn.Embedding(self.n_ent, self.dims)).to(device)
        self.relations = (nn.Embedding(self.n_rel, self.dims)).to(device)
        self.w_relations = (nn.Embedding(self.n_rel, self.dims)).to(device)

        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.relations.weight.data,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.w_relations.weight.data,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))

        self.relations.weight.data = nn.functional.normalize(self.relations.weight.data, dim = -1)
        self.w_relations.weight.data = nn.functional.normalize(self.w_relations.weight.data, dim = -1)

    def distance(self, h, r, t, w_r):
        
        #print (torch.sum(h*w_r, dim = -1, keepdim = True).repeat(1, self.dims).shape, w_r.shape)
        ht = h - torch.sum(h*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        tt = t - torch.sum(t*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r

        return -torch.pow(torch.norm(ht + r - tt, dim = -1, p = 2),2).to(device)


    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)


        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])
        
        w_pred = self.w_relations(data[:, 2])
        w_pred = nn.functional.normalize(w_pred, dim = -1)


        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        
        #print (self.distance(head, pred, tail).shape)


        return self.distance(head, pred, tail, w_pred), self.distance(cHead, pred, cTail, w_pred), t
