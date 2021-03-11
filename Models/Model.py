import torch
import torch.nn as nn
from .BaseModule import BaseModule


class Model(BaseModule):

    def __init__(self, ent_tot, rel_tot):
        super(Model, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def checkForPositives(self, data):
        pos = 0
        neg = 0

        for i in range(len(data)):
            if data[i]>=0:
                pos+=1
            else:
                neg+=1

        print (pos, neg)