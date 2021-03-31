import torch
import torch.nn as nn
from .Model import Model

class SimplE(Model):

    def __init__(self, ent_tot, rel_tot, dim = 100):
        super(SimplE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.ent_h_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_t_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_h_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_t_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    def _calc_avg(self, h_i, t_i, h_j, t_j, r, r_inv):
        return (torch.sum(h_i * r * t_j, -1) + torch.sum(h_j * r_inv * t_i, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def startingBatch(self):
        # Do nothing
        return

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h_i = self.ent_h_embeddings(batch_h)
        h_j = self.ent_h_embeddings(batch_t)
        t_i = self.ent_t_embeddings(batch_h)
        t_j = self.ent_t_embeddings(batch_t)

        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        
        score = self._calc_avg(h_i, t_i,h_j, t_j, r, r_inv).flatten()
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 4
        return regul

    def predict(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_h_embeddings(batch_h)
        t = self.ent_t_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = -self._calc_ingr(h, r, t)
        return score