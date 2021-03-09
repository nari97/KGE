import torch
import torch.nn as nn
from .Model import Model
import torch.nn.functional as F
class DistMult(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=None, epsilon=None):
        super(DistMult, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

    def _calc(self, h, t, r, mode):
        h = F.normalize(h, p = 2, dim = -1)
        r = F.normalize(r, p = 2, dim = -1)
        t = F.normalize(t, p = 2, dim = -1)
        score = (h * r) * t
        score = torch.sum(score, -1).flatten()
        return score

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
        score = self._calc(h, t, r, mode)
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def l3_regularization(self):
        return (self.ent_embeddings.weight.norm(p=3) ** 3 + self.rel_embeddings.weight.norm(p=3) ** 3)

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu()
