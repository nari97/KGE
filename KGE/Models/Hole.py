import torch
import torch.nn as nn
import numpy as np
import time

device = torch.device("cuda:0")

class Hole(nn.Module):

    def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(Hole, self).__init__()
        self.name = 'Hole'
        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm

        self.entities = (nn.Embedding(self.n_ent, self.dims)).to(device)
        self.relations = (nn.Embedding(self.n_rel, self.dims)).to(device)

        self.loss = nn.MarginRankingLoss(margin = self.margin, reduction = 'mean')
        self.loss = self.loss.to(device)
        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.relations.weight.data,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))

        self.relations.weight.data = nn.functional.normalize(self.relations.weight.data, dim = 1)

    def distance(self, h, r, t):
        
        fourierH = torch.rfft(h, signal_ndim = 1, onesided = False).to(device)
        fourierT = torch.rfft(t, signal_ndim = 1, onesided = False).to(device)

        conjH = torch.conj(torch.view_as_complex(fourierH)).to(device)
        complexT = torch.view_as_complex(fourierT).to(device)

        fInvF = torch.ifft(torch.view_as_real(torch.mul(conjH, complexT)), signal_ndim = 1).to(device)
        fInvF = torch.view_as_complex(fInvF).to(device)
        
        real = fInvF.real
        real.to(device)
        answer = []

        for i in range(len(r)):
          ri = torch.reshape(r[i], (1, self.dims))
          
          reali = torch.reshape(real[i], (self.dims, 1))

          ans = torch.matmul(ri, reali)
          answer.append(ans)
          
        
        

        answer = torch.Tensor(answer)
        answer.requires_grad = True

        return answer.to(device)


    def forward(self, data):

        t = torch.ones((len(data), 1))
        t = torch.Tensor(-t)
        t = t.to(device)

        data = torch.LongTensor(data)
        data = data.to(device)

        head = self.entities(data[:, 0])
        tail = self.entities(data[:, 1])
        pred = self.relations(data[:, 2])

        cHead = self.entities(data[:, 3])
        cTail = self.entities(data[:, 4])

        
        #print (self.distance(head, pred, tail).shape)


        return self.loss(self.distance(head, pred, tail), self.distance(cHead, pred, cTail), t)
