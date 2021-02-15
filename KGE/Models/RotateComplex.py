import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.fft

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RotateComplex(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(RotateComplex, self).__init__()
        self.name = 'Rotate'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm
        self.abs = torch.Tensor([1]).to(device)

        self.entities = torch.zeros((n_ent, dims, 2))
        self.relations = torch.zeros((n_rel, 1))
        

        nn.init.uniform_(self.entities, a = -(6/np.sqrt(self.dims)), b = (6/np.sqrt(self.dims)))
        nn.init.uniform_(self.relations, a = - (6/np.sqrt(self.dims)), b = (6/np.sqrt(self.dims)))

        self.entities = nn.Parameter(self.entities, requires_grad = True)
        self.entities.to(device)
        self.relations = nn.Parameter(self.relations, requires_grad = True)
        self.relations.to(device)
    

    def distance(self, h,r,t):
      th = torch.view_as_complex(h)
      tr = torch.polar(self.abs, r)
      tt = torch.view_as_complex(t)
      data = -torch.linalg.norm(th*tr - tt, ord = 1)
      
      return data.to(device)

    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        #data = torch.LongTensor(data)
        
        #data = data.to(device)

        head = self.entities[data[:, 0]]
        tail = self.entities[data[:, 1]]
        pred = self.relations[data[:, 2]]

        cHead = self.entities[data[:, 3]]
        cTail = self.entities[data[:, 4]]

        #print (pred[0:10])
        ps = torch.sigmoid(self.distance(head, pred, tail))
        ns = torch.sigmoid(self.distance(cHead, pred, cTail))

        return ps, ns, t
