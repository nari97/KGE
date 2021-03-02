import torch
import torch.nn as nn
from .Model import Model


class RotatE(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=None):
        super(RotatE, self).__init__(ent_tot, rel_tot)
        self.dim = dim
        self.dim_e = dim * 2
        self.dim_r = dim
        print (self.dim_e, self.dim_r)
        self.margin = margin
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        nn.init.uniform_(self.ent_embeddings.weight.data)
        nn.init.uniform_(tensor=self.rel_embeddings.weight.data, a=0, b=2*self.pi_const.item())

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False


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

    def _calc(self, h, t, r, mode):
        
        th = h.view(h.shape[0], self.dim, -1)
        tt = t.view(t.shape[0], self.dim, -1)      
      
        real = torch.cos(r)
        img = torch.sin(r)

        tr = torch.stack((real, img), dim = -1)
      
        data = -torch.norm(self.multiply(th, tr) - tt, dim = 1, p = 2)

        return data.flatten()
    

    def startingBatch(self):
        # Do nothing
        return

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        #print ('r', r.shape)
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def predict(self, data):
        score = -self.forward(data)
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul
