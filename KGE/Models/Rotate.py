import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.fft

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Rotate(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(Rotate, self).__init__()
        self.name = 'Rotate'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm
        

        self.entities = nn.Embedding(self.n_ent, self.dims*2).to(device)
        self.relations = nn.Embedding(self.n_rel, self.dims).to(device)
        

        nn.init.uniform_(self.entities.weight.data, a = -(6/np.sqrt(self.dims*2)), b = (6/np.sqrt(self.dims*2)))
        nn.init.uniform_(self.relations.weight.data, a = - (6/np.sqrt(self.dims)), b = (6/np.sqrt(self.dims)))
        
      
    def multiply(self, a, b):
      #print (a.shape)
      x = a[:,:,0]
      y = a[:,:,1]
      u = b[:,:,0]
      v = b[:,:,1]

      a = x*u - y*v
      b = x*v + y*u
      c = torch.stack((a,b), dim = -1)
      
      return c

    def distance(self, h,r,t):
    
      th = h.view(h.shape[0], self.dims, -1)
      tt = t.view(t.shape[0], self.dims, -1)
      
      
      #tr = r/((6/np.sqrt(self.dims))/math.pi)
      
      real = torch.cos(r)
      img = torch.sin(r)

      tr = torch.stack((real, img), dim = -1)
      
      data = -torch.norm(self.multiply(th, tr) - tt, dim = 1, p = 2)
      
      return data.to(device)

    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        #data = torch.LongTensor(data)
        
        #data = data.to(device)

        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])

        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        #print (pred[0:10])
        ps = torch.sigmoid(self.distance(head, pred, tail))
        ns = torch.sigmoid(self.distance(cHead, pred, cTail))

        return ps, ns, t
