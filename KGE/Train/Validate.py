import torch
import torch.nn as nn
import time
import os

torch.multiprocessing.freeze_support()
torch.no_grad()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def corruptSubjectAndObjects(s, p, o, data):
    cs = set()
    co = set()
    for row in data:
        if row[1] == o and row[2] == p:
            
            pass
        else:
            cs.add(row[0])
            
    for row in data:
        if row[0] == s and row[2] == p:            
            pass
        else:
            co.add(row[1])

    return torch.LongTensor(list(cs)).to(device), torch.LongTensor(list(co)).to(device)

def mr(r):
  mr = 0

  for rz in r:
    mr += rz

  mr /= len(r)
  return mr.item()

def gmr(r):
  mr = 1

  for rz in r:
    mr *= rz

  mr /= len(r)
  return mr.item()

def validate(loader, modelName, datasetName, verbose = False):
    folder = os.getcwd() + "/Datasets/" + datasetName
    loader = loader

    entities = loader.getEntities()
    relations = loader.getRelations()
    dataset = loader.getTrainDataset()
    valid = loader.getValidDataset()
    
    f = open(os.getcwd() + "/SavedModels//" + modelName + ".pkl", 'rb')
    model = torch.load(f)
    #print (model.name)
    f.close()

    finalDataset = dataset + valid
    
    valid = valid[:100]

    if verbose:
        print ('GY size : ',len(finalDataset))

    r = []

     
    for i, row in enumerate(valid):
        before = time.time()
        rless = 1
        req = 0
        
        cs, co = corruptSubjectAndObjects(row[0], row[2], row[1], finalDataset)
        
        d1 = d2 = d3 = None

        s = torch.LongTensor([row[0]]).to(device)
        p = torch.LongTensor([row[2]]).to(device)
        o = torch.LongTensor([row[1]]).to(device)
        w_r = None
        m_r = None

        sEmb = model.entities(s)
        pEmb = model.relations(p)
        oEmb = model.entities(o)

        if 'TransH' in modelName:
          w_r = model.w_relations(p)
        if 'TransR' in modelName:
          m_r = model.mr(p)

        if 'TransH' in modelName:
          d1 = model.distance(sEmb, pEmb, oEmb, w_r)
        elif 'TransR' in modelName:
          d1 = model.distance(sEmb, pEmb, oEmb, m_r)
        else:
          d1 = model.distance(sEmb, pEmb, oEmb)

        
        pEmb = model.relations(p).repeat(len(cs), 1)
        if 'TransH' in modelName:
          wrEmb = w_r.repeat(len(cs), 1)
        elif 'TransR' in modelName:
          mrEmb = m_r.repeat(len(cs), 1)

        oEmb = model.entities(o).repeat(len(cs), 1)
        sDashEmb = model.entities(cs)
        
        if 'TransH' in modelName:
          d2 = model.distance(sDashEmb, pEmb, oEmb, wrEmb)
        elif 'TransR' in modelName:
          d2 = model.distance(sDashEmb, pEmb, oEmb, mrEmb)
        else:
          d2 = model.distance(sDashEmb, pEmb, oEmb)

        rl = d1>d2
        re = (d1 == d2)

        rless += torch.sum(rl)
        req = torch.sum(re)
        
        r.append((2*rless + req)/2)
        
        sEmb = model.entities(s).repeat(len(co), 1)
        pEmb = model.relations(p).repeat(len(co), 1)
        oDashEmb = model.entities(co)
        
        if 'TransH' in modelName:
          wrEmb = w_r.repeat(len(co), 1)
        elif 'TransR' in modelName:
          mrEmb = m_r.repeat(len(co), 1)

        if 'TransH' in modelName:
          d3 = model.distance(sEmb, pEmb, oDashEmb, wrEmb)
        elif 'TransR' in modelName:
          d3 = model.distance(sEmb, pEmb, oDashEmb, mrEmb)
        else:
          d3 = model.distance(sEmb, pEmb, oDashEmb)
        
        rless = 1
        req = 0
        
        
        rl = d1>d3
        re = (d1 == d3)

        rless += torch.sum(rl)
        req = torch.sum(re)
        

        if verbose:
          if i%100 == 0:
            print ('Row : ', i)

        r.append((2*rless  +req)/2)
        after = time.time()
             
    return mr(r)