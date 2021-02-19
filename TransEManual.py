import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import Model

class TransEManual(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, batch_size = 256, alpha=1.0, mode=None,
                 norm_flag=True, seed=None, normalize_r=True, norm_start=False):
        super(TransEManual, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.batch_size = batch_size
        self.alpha = alpha
        self.mode = mode
        self.norm_flag = norm_flag

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        if seed is not None:
            torch.manual_seed(seed)

        self.normalize_r = normalize_r
        self.norm_start = norm_start
        self.loss = 0
        self.data = None

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self, data):
        self.data = data
        return self

    def backward(self):
        batch_h_pos = self.ent_embeddings(self.data['batch_h'][:self.batch_size])
        batch_t_pos = self.ent_embeddings(self.data['batch_t'][:self.batch_size])
        batch_r_pos = self.rel_embeddings(self.data['batch_r'][:self.batch_size])

        batch_h_neg = self.ent_embeddings(self.data['batch_h'][self.batch_size:])
        batch_t_neg = self.ent_embeddings(self.data['batch_t'][self.batch_size:])
        batch_r_neg = self.rel_embeddings(self.data['batch_r'][self.batch_size:])

        # double z1 = -learning_rate, z2 = learning_rate;
        jRange = batch_h_neg.size()[0] // self.batch_size
        z1, z2 = -self.alpha, self.alpha

        if self.norm_flag:
            batch_h_pos = F.normalize(batch_h_pos, 2, -1)
            batch_h_neg = F.normalize(batch_h_neg, 2, -1)
            if self.normalize_r:
                batch_r_pos = F.normalize(batch_r_pos, 2, -1)
                batch_r_neg = F.normalize(batch_r_neg, 2, -1)
            batch_t_pos = F.normalize(batch_t_pos, 2, -1)
            batch_t_neg = F.normalize(batch_t_neg, 2, -1)

        if self.mode == "pge":
            # double x1 = 2 * (h_vec[i] + r_vec[i] - t_vec[i]);
            # double x2 = 2 * (ch_vec[i] + r_vec[i] - ct_vec[i]);
            batch_pos, batch_neg = 2 * (batch_h_pos + batch_r_pos - batch_t_pos), \
                                   2 * (batch_h_neg + batch_r_neg - batch_t_neg)

            for j in range(jRange):
                batch_h_neg_j = batch_h_neg[j * self.batch_size:(j + 1) * self.batch_size]
                batch_t_neg_j = batch_t_neg[j * self.batch_size:(j + 1) * self.batch_size]
                batch_neg_j = batch_neg[j * self.batch_size:(j + 1) * self.batch_size]

                # r_vec[i] += z1 * x1 + z2 * x2;
                # h_vec[i] += z1 * x1;
                # t_vec[i] += -z1 * x1;
                # ch_vec[i] += z2 * x2;
                # ct_vec[i] += -z2 * x2;

                batch_r_pos += z1 * batch_pos + z2 * batch_neg_j
                batch_h_pos += z1 * batch_pos
                batch_t_pos += -z1 * batch_pos
                batch_h_neg_j += z2 * batch_neg_j
                batch_t_neg_j += -z2 * batch_neg_j
        elif self.mode == "ours":
            batch_pos, batch_neg = (batch_h_pos + batch_r_pos - batch_t_pos)/F.normalize(batch_h_pos + batch_r_pos - batch_t_pos, 2, -1), \
                                   (batch_h_neg + batch_r_neg - batch_t_neg)/F.normalize(batch_h_neg + batch_r_neg - batch_t_neg, 2, -1)

            # Set NaNs to zero
            batch_pos[batch_pos != batch_pos] = 0
            batch_neg[batch_neg != batch_neg] = 0

            for j in range(jRange):
                batch_h_neg_j = batch_h_neg[j*self.batch_size:(j+1)*self.batch_size]
                batch_t_neg_j = batch_t_neg[j*self.batch_size:(j+1)*self.batch_size]
                batch_neg_j = batch_neg[j*self.batch_size:(j+1)*self.batch_size]

                pos_h, pos_t = self.data['batch_h'][:self.batch_size],self.data['batch_t'][:self.batch_size]
                neg_h, neg_t = self.data['batch_h'][(j+1)*self.batch_size:(j+2)*self.batch_size], \
                               self.data['batch_t'][(j+1)*self.batch_size:(j+2)*self.batch_size]

                diffh, difft = pos_h != neg_h, pos_t != neg_t
                diffh, difft = torch.Tensor(self.dim*[1]) * diffh.reshape(-1,1), \
                               torch.Tensor(self.dim*[1]) * difft.reshape(-1,1)

                batch_r_pos += -self.alpha * (batch_pos - batch_neg_j)
                # Head was corrupted.
                batch_h_pos += -self.alpha * diffh * (batch_pos)
                batch_t_pos += -self.alpha * diffh * (batch_neg_j - batch_pos)
                batch_h_neg_j += -self.alpha * diffh * (-batch_neg_j)
                # Tail was corrupted.
                batch_h_pos += -self.alpha * difft * (batch_pos - batch_neg_j)
                batch_t_pos += -self.alpha * difft * (-batch_pos)
                batch_t_neg_j += -self.alpha * difft * (batch_neg_j)

        for i in range(self.batch_size):
            self.ent_embeddings.weight.data[self.data['batch_h'][i]] = batch_h_pos[i]
            self.ent_embeddings.weight.data[self.data['batch_t'][i]] = batch_t_pos[i]
            self.rel_embeddings.weight.data[self.data['batch_r'][i]] = batch_r_pos[i]
            for j in range(jRange):
                self.ent_embeddings.weight.data[self.data['batch_h'][i+(j+1)*self.batch_size]] = batch_h_neg[i+j*self.batch_size]
                self.ent_embeddings.weight.data[self.data['batch_t'][i+(j+1)*self.batch_size]] = batch_t_neg[i+j*self.batch_size]

        # Compute loss
        batch_h_pos = self.ent_embeddings(self.data['batch_h'][:self.batch_size])
        batch_t_pos = self.ent_embeddings(self.data['batch_t'][:self.batch_size])
        batch_r_pos = self.rel_embeddings(self.data['batch_r'][:self.batch_size])

        batch_h_neg = self.ent_embeddings(self.data['batch_h'][self.batch_size:])
        batch_t_neg = self.ent_embeddings(self.data['batch_t'][self.batch_size:])
        batch_r_neg = self.rel_embeddings(self.data['batch_r'][self.batch_size:])

        batch_pos, batch_neg = batch_h_pos + batch_r_pos - batch_t_pos, batch_h_neg + batch_r_neg - batch_t_neg

        scores = None
        for j in range(jRange):
            batch_neg_j = batch_neg[j * self.batch_size:(j + 1) * self.batch_size]
            theseScores = torch.max((torch.norm(batch_pos, 2, -1) - torch.norm(batch_neg_j, 2, -1)),
                                    torch.Tensor([-self.margin] * self.batch_size))
            if scores is None:
                scores = theseScores
            else:
                scores = torch.cat((scores, theseScores), -1)
        self.loss = (scores.mean() + self.margin).item()

    def item(self):
        return self.loss

    def startingBatch(self):
        if self.norm_start:
            self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
        return


