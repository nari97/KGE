import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class TransR(nn.Module):

    def __init__(self, n_ent, n_rel, dimsE, dimsR, margin, norm):
        super(TransR, self).__init__()
        self.name = 'TransR'
        self.dimsE = dimsE
        self.dimsR = dimsR
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm

        self.entities = (nn.Embedding(self.n_ent, self.dimsE)).to(device)
        self.relations = (nn.Embedding(self.n_rel, self.dimsR)).to(device)
        self.mr = (nn.Embedding(self.n_rel, self.dimsE*dimsR)).to(device)

        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dimsE), b = 6/np.sqrt(self.dimsE))
        nn.init.uniform_(self.relations.weight.data,  a = -6/np.sqrt(self.dimsR), b = 6/np.sqrt(self.dimsR))
        nn.init.uniform_(self.mr.weight.data,  a = -6/np.sqrt(self.dimsE*self.dimsR), b = 6/np.sqrt(self.dimsE*self.dimsR))

        self.relations.weight.data = nn.functional.normalize(self.relations.weight.data, dim = 1)

    def distance(self, h, r, t, m):
        m = m.view(m.shape[0], self.dimsE, self.dimsR)

        hr = h.repeat((1, 1, m.shape[-1])).reshape(h.shape[0], h.shape[1], m.shape[-1])
        tr = t.repeat((1, 1, m.shape[-1])).reshape(t.shape[0], t.shape[1], m.shape[-1])

        hr = (hr*m).sum(dim = 1)
        tr = (tr*m).sum(dim = 1)

        hr = nn.functional.normalize(hr, dim = -1)
        tr = nn.functional.normalize(tr, dim = -1)

        return torch.pow(torch.norm(hr+r-tr, dim = 1, p = 2),2)


    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        
        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])

        mr = self.mr(data[:, 2])
        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        return self.distance(head, pred, tail, mr), self.distance(cHead, pred, cTail, mr), t
