import torch
import torch.nn as nn
import numpy as np
import time
import torch.fft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Hole(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(Hole, self).__init__()
        self.name = 'Hole'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm
        self.ents = torch.LongTensor([i for i in range (0, self.n_ent)]).to(device)
        self.entities = (nn.Embedding(self.n_ent, self.dims)).to(device)
        self.relations = (nn.Embedding(self.n_rel, self.dims)).to(device)

        
        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.relations.weight.data,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))

        self.relations.weight.data = nn.functional.normalize(self.relations.weight.data, dim = 1)

    def createHkTk(self):
        
        
        entities = self.entities(self.ents)
        tks = torch.cat([torch.cat([entities[i].roll(k) for k in range(entities.shape[1])]).view(entities.shape[1], entities.shape[1]) for i in range(entities.shape[0])]).view(entities.shape[0], entities.shape[1], entities.shape[1])
        his = entities.repeat(1, entities.shape[1]).view(entities.shape[0], entities.shape[1], -1)
        tks = tks.to(device)
        his = his.to(device)

        return his, tks

    def multiply(self, a, b):

      x = a[:,:,0]
      y = a[:,:,1]
      u = b[:,:,0]
      v = b[:,:,1]

      ans = torch.zeros(a.shape[0], a.shape[1], 2)

      ans[:,:,0] = x*u - y*v
      ans[:,:,1] = x*v + y*u

      return ans.to(device)

    def distanceHole(self, h, r, t):
        
        
      
      tks = torch.cat([torch.cat([t[i].roll(k) for k in range(t.shape[1])]).view(t.shape[1], t.shape[1]) for i in range(t.shape[0])]).view(t.shape[0], t.shape[1], t.shape[1])
      his = h.repeat(1, h.shape[1]).view(h.shape[0], h.shape[1], -1)
      tks = tks.to(device)
      his = his.to(device)

      

      m = (his*tks).sum(dim = 2) 
      m = m.to(device)
        
      answer = torch.sum(r*m, dim = 1)
      

      return answer

        

    def distance(self, h,r,t):

      
      fourierH = torch.fft.rfft(h, dim = -1)
      fourierT = torch.fft.rfft(t, dim = -1)
      
      conjH = torch.conj(fourierH)
        
      inv = torch.fft.irfft(conjH*fourierT, dim = -1)
      
      answer = torch.sum(r*inv, dim = 1)
      #answer.requires_grad = True

      return answer.to(device)

    def forwardMethod2(self, data):
        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        #data = torch.LongTensor(data)
        data = data.to(device)

        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])

        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        #print ('Head, tail, pred fn2 : ', head, tail, pred)
        
        #print (self.distance(head, pred, tail).shape)


        return torch.sigmoid(self.distanceHole(head, pred, tail)), torch.sigmoid(self.distanceHole(cHead, pred, cTail)), t

    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        #data = torch.LongTensor(data)
        data = data.to(device)

        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])

        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        #print (cHead.device)
        #print ('Head, tail, pred fn1 : ', head, tail, pred)
        
        #print (self.distance(head, pred, tail).shape)


        ps = torch.sigmoid(self.distance(head, pred, tail))
        ns = torch.sigmoid(self.distance(cHead, pred, cTail))

        return ps, ns, t
