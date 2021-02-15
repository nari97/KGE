import torch
from torch import nn as nn
import numpy as np
from Train.Validate import validate
import time
import os

#torch.multiprocessing.freeze_support()
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer:

    def __init__(self, model, datasetName, loader, margin = 1, batch_size = 10000, epochs = 1000, nr = 1, lr = 0.01, weight_decay = 0, normalize = True, momentum = 0):
        self.model = model
        self.datasetName = datasetName
        self.loader = loader
        self.margin = margin
        self.batch_size = batch_size
        self.epochs = epochs
        self.nr = nr
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalize = normalize
        self.momentum = momentum

    def corrupt(self, data, entities):
        
        corruptedData = []
        for i in range(0, len(data)):
            for j in range(self.nr):
                x = np.random.randint(0,2)
                if x == 0:
                    cH = data[i][0]
                    while cH==data[i][0]:
                      cH = np.random.randint(0, len(entities))
                    row = [data[i][0], data[i][1], data[i][2], cH, data[i][1]]
                else:
                    cT = data[i][1]
                    while cT == data[i][1]:
                      cT = np.random.randint(0, len(entities))
                    row = [data[i][0], data[i][1], data[i][2], data[i][0], cT]
                corruptedData.append(row)

        return torch.LongTensor(corruptedData).to(device)

    
    def train(self):
        
        modelName = self.model.name
        datasetName = self.datasetName
        modelName = modelName + "_" + datasetName
        patient = 3
        patient_count = 0
        prev_mr = 9999999
        mr = 0

        
        entities = self.loader.getEntities()
        relations = self.loader.getRelations()
        dataset = self.loader.getTrainDataset()
        
        print ('Dataset : ', datasetName)
        print ('Model : ', modelName)

        print ('Dataset shape : ', len(dataset))

        corruptedDataset = self.corrupt(dataset, entities)

        print ('After corruption : ', corruptedDataset.shape)  
        
        traindata = torch.utils.data.DataLoader(corruptedDataset, batch_size = int(len(corruptedDataset)/self.batch_size), shuffle = False, num_workers = 0)
        
        print('Data batched')
        lossFn = nn.MarginRankingLoss(margin = self.margin, reduction = 'mean')
        lossFn = lossFn.to(device)
        
        self.model = self.model.to(device)
        optim = torch.optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay, momentum = self.momentum)
        
        for epoch in range(1,self.epochs+1):
            before = time.time()
            
            if self.normalize:
              self.model.entities.weight.data = nn.functional.normalize(self.model.entities.weight.data, dim = -1)
              if 'TransH' in self.model.name:
                self.model.w_relations.weight.data = nn.functional.normalize(self.model.w_relations.weight.data, dim = -1)
              if 'TransR' in self.model.name:
                self.model.relations.weight.data = nn.functional.normalize(self.model.relations.weight.data, dim = -1)
            
            for i, data in enumerate(traindata):
              
              optim.zero_grad()
              ps, ns, t = self.model.forward(data)
              loss = lossFn(ps, ns, t)
              loss.backward()
              optim.step()
                
                
            after = time.time()
            print ('Epoch :', epoch, " : ", after - before)
            

            if epoch%10 == 1:
                f = open(os.getcwd() + '/SavedModels//' + modelName + '.pkl', 'wb')
                torch.save(self.model, f)
                f.close()
                before = time.time()
                mr = validate(modelName = modelName, datasetName = datasetName, loader = self.loader)
                after = time.time()

                print ('MR : ', mr, ': ', after-before)

                if mr>prev_mr:
                    patient_count+=1
                    if patient_count>patient:
                      break
                else:
                    prev_mr = mr
                    patient_count = 0
                    f = open(os.getcwd() + '/SavedModels//' + modelName + 'Best.pkl', 'wb')
                    torch.save(self.model, f)
                    f.close()

        return prev_mr