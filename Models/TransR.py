import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransR(Model):

    def __init__(self, ent_tot, rel_tot, dim_e=100, dim_r = 100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransR, self).__init__(ent_tot, rel_tot)

        self.dimsE = dim_e
        self.dimsR = dim_r
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dimsE)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dimsR)
        self.norm_vector = nn.Embedding(self.rel_tot, self.dimsE*self.dimsR)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.norm_vector.weight.data)
        else:
            self.embedding_range_entities = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dimsE]), requires_grad=False
            )
            self.embedding_range_relations = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dimsR]), requires_grad=False
            )
            self.embedding_range_m = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / (self.dimsE*self.dimsR)]), requires_grad=False
            )

            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range_entities.item(),
                b=self.embedding_range_entities.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range_relations.item(),
                b=self.embedding_range_relations.item()
            )
            nn.init.uniform_(
                tensor=self.norm_vector.weight.data,
                a=-self.embedding_range_m.item(),
                b=self.embedding_range_m.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, m, mode):
        
        m = m.view(m.shape[0], self.dimsE, self.dimsR)

        hr = h.repeat((1, 1, m.shape[-1])).reshape(h.shape[0], h.shape[1], m.shape[-1])
        tr = t.repeat((1, 1, m.shape[-1])).reshape(t.shape[0], t.shape[1], m.shape[-1])

        hr = (hr*m).sum(dim = 1)
        tr = (tr*m).sum(dim = 1)

        hr = nn.functional.normalize(hr, dim = -1)
        tr = nn.functional.normalize(tr, dim = -1)
        answer = torch.pow(torch.norm(hr+r-tr, dim = 1, p = 2),2)
        return answer.flatten()


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
        r_norm = self.norm_vector(batch_r)
        
        score = self._calc(h, t, r, r_norm, mode)
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
        r_norm = self.norm_vector(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(r_norm ** 2)) / 4
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score
        else:
            return score