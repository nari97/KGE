from Models.TransE import TransE
from Train.Trainer import Trainer
from Loader.Loader import Loader

datasetName = 'WN18'
dims = 180
batch_size = 10000
epochs = 1000
nr = 21
norm = 1
lr = 0.43968548804984126
margin = 7.826514170133984
weight_decay = 2.4439974195366027e-07

loader = Loader(datasetName)
n_ent, n_rel = loader.getEntityAndRelationCount()

model = TransE(n_ent, n_rel, dims, margin, norm)

print (model.name)

trainer = Trainer(model, datasetName, loader, margin = margin, batch_size = batch_size, epochs = epochs, nr = nr, lr = lr, weight_decay = weight_decay)

trainer.train()
