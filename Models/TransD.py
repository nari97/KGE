import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransD(Model):

    def __init__(self, ent_tot, rel_tot, dim_e=100, dim_r=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransD, self).__init__(ent_tot, rel_tot)

        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
            )
            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.ent_transfer.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_transfer.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            #h = F.normalize(h, 2, -1)
            r = F.normalize(r, p = 2, dim = -1)
            #t = F.normalize(t, 2, -1)
        #print(h.shape, r.shape, t.shape)
        score = h + r - t

        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def _transfer(self, e, e_transfer, r_transfer):
        
        e = F.normalize(e, p = 2, dim =-1)

        wr = r_transfer.T
        wh = e_transfer
        et = e.T

        mat = torch.matmul(torch.matmul(wr, wh), et)

        return F.normalize(mat.T, p=2, dim=-1)

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

        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)

        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        
        score = self._calc(h, t, r, mode)

        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(h_transfer ** 2) +
                 torch.mean(t_transfer ** 2) +
                 torch.mean(r_transfer ** 2)) / 6
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score
        else:
            return score