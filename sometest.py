import torch
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import wilcoxon
from DataLoader.DataLoader import DataLoader

loader = DataLoader("Datasets/YAGO3-10/", type = "train")

buckets = {"0.8" : [], "0.6" : [], "0.4" : [], "0.2" : [], "0" : []}
counts = {}
anomaly = loader.relation_anomaly
for triple in loader.getTriples():
    r_anom = anomaly[triple.r]
    if triple.r not in counts:
        counts[triple.r] = 1
    else:
        counts[triple.r]+=1
    #print (r_anom)
    if r_anom>=0.8:
        buckets["0.8"].append(triple)
    elif r_anom>=0.6:
        buckets["0.6"].append(triple)
    elif r_anom>=0.4:
        buckets["0.4"].append(triple)
    elif r_anom>=0.2:
        buckets["0.2"].append(triple)
    else:
        buckets["0"].append(triple)

print (len(loader.getTriples()))

for key in buckets.keys():
    print(len(buckets[key]))

print (counts)