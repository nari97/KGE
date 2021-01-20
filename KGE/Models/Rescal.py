import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0")

class Rescal(nn.Module):
  
  def __init__(self, n_ent, n_rel, dims, margin, norm):
        super(Rescal, self).__init__()

        self.dims = dims
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.margin = margin
        self.norm = norm

        self.entities = (nn.Embedding(self.n_ent, self.dims)).to(device)
        self.relations = torch.zeros([n_rel, dims, dims], device = device, requires_grad = True)

        self.loss = nn.MarginRankingLoss(margin = self.margin, reduction = 'mean')
        self.loss = self.loss.to(device)
        nn.init.uniform_(self.entities.weight.data, a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))
        nn.init.uniform_(self.relations,  a = -6/np.sqrt(self.dims), b = 6/np.sqrt(self.dims))

  def distance(self, h, r ,t):
    vals = []

    for i in range(len(h)):
      #hi = torch.reshape(h[i], (1, h[i].shape[0]))
      #ti = torch.reshape(t[i], (1, t[i].shape[0])).t
      val = torch.matmul(torch.matmul(h[i],r[i]),t[i])
      print (val.shape)
      #print (h[i].shape, r[i].shape, torch.t(t[i]).shape, val.shape)
      vals.append(val.item())

    #print (len(vals) , vals)
    return torch.Tensor(vals).to(device)

  def forward(self, data):

    t = torch.ones((len(data), 1))
    t = torch.Tensor(-t)
    t = t.to(device)

    data = torch.LongTensor(data)
    data = data.to(device)

    head = self.entities(data[:, 0])
    tail = self.entities(data[:, 1])

    pred = self.relations[data[:, 2]]

    cHead = self.entities(data[:, 3])
    cTail = self.entities(data[:, 4])

    return self.loss(self.distance(head, pred, tail), self.distance(cHead, pred, cTail), t)



        