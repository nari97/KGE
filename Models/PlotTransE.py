import torch
# Need to make the 3D work!
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss
from TransE import TransE
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

def plotVectors():
    id2Entity = {0:'MJFox', 1: 'CLloyd', 2: 'CWells', 3:'EShue', 4: 'MaryS', 5: 'BFI', 6: 'BFII', 7: 'BFIII',
                 8: 'KKid', 9: 'TheHelp', 10: 'McFly', 11: 'DrBrown', 12: 'Jenn'}
    id2Relation = {0: 'actsIn', 1: 'playedBy'}

    entityTotal = 13
    relTotal = 2

    train_h = [0, 0, 0, 10, 1, 1, 11, 3, 3, 3, 12, 12, 4, 4]
    train_r = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    train_t = [5, 6, 7, 0, 6, 7, 1, 6, 7, 8, 2, 3, 7, 9]
    batchSize = len(train_h)

    #for i in range(batchSize):
    #    print("(", id2Entity[batch_h[i]], ",", id2Relation[batch_r[i]], ",", id2Entity[batch_t[i]], ") -> ", batch_y[i])
    #    print("(", id2Entity[batch_h[i+batchSize]], ",", id2Relation[batch_r[i+batchSize]], ",", id2Entity[batch_t[i+batchSize]], ") -> ", batch_y[i+batchSize])
    #    print("(", id2Entity[batch_h[i+2*batchSize]], ",", id2Relation[batch_r[i+2*batchSize]], ",", id2Entity[batch_t[i+2*batchSize]], ") -> ", batch_y[i+2*batchSize])
    #    print()

    transx = TransE(
        ent_tot=entityTotal,
        rel_tot=relTotal,
        dim=3,
        p_norm=2,
        norm_flag=True)

    train(transx, batchSize, train_h, train_t, train_r, 10000)

    test = []
    for h in range(entityTotal):
        for r in range(relTotal):
            for t in range(entityTotal):
                if present(train_h, train_r, train_t, h, r, t):
                    continue
                p = transx.predict({
                    'batch_h': to_var(np.array([h], dtype=np.int64)),
                    'batch_t': to_var(np.array([t], dtype=np.int64)),
                    'batch_r': to_var(np.array([r], dtype=np.int64)),
                    'batch_y': to_var(np.array([1], dtype=np.int32)),
                    'mode': 'normal'
                })
                test = test + [{'triple':[h, r, t], 'score':p}]
    test = sorted(test, key=lambda k: k['score'])

    print("Top 10")
    for i in range(10):
        print("(", id2Entity[test[i]['triple'][0]], ",", id2Relation[test[i]['triple'][1]], ",",
              id2Entity[test[i]['triple'][2]], ") ->", test[i]['score'])
    print("Bottom 10")
    for i in range(len(test)-1,len(test)-10,-1):
        print("(", id2Entity[test[i]['triple'][0]], ",", id2Relation[test[i]['triple'][1]], ",",
              id2Entity[test[i]['triple'][2]], ") ->", test[i]['score'])

    # Missing: CLloyd, actsIn, BFI
    h, r, t = 1, 0, 5
    plotTriple(transx, h, r, t, id2Entity, id2Relation)
    # Missing: CWells, actsIn, BFI
    h, r, t = 2, 0, 5
    plotTriple(transx, h, r, t, id2Entity, id2Relation)
    # Incorrect: EShue, actsIn, BFI
    h, r, t = 3, 0, 5
    plotTriple(transx, h, r, t, id2Entity, id2Relation)

    # Let's predict:
    print("Prediction:", transx.predict({
        'batch_h': to_var(np.array([1, 2, 3], dtype=np.int64)),
        'batch_t': to_var(np.array([5, 5, 5], dtype=np.int64)),
        'batch_r': to_var(np.array([0, 0, 0], dtype=np.int64)),
        'batch_y': to_var(np.array([1, 1, 1], dtype=np.int32)),
        'mode': 'normal'
    }))

def plotTriple(transx, h, r, t, id2Entity, id2Relation):
    # Let's visualize
    h_emb = transx.ent_embeddings(torch.LongTensor([h]))
    r_emb = transx.rel_embeddings(torch.LongTensor([r]))
    t_emb = transx.ent_embeddings(torch.LongTensor([t]))

    h_emb = F.normalize(h_emb, 2, -1)
    r_emb = F.normalize(r_emb, 2, -1)
    t_emb = F.normalize(t_emb, 2, -1)

    h_flat = h_emb.data.numpy().flatten()
    r_flat = r_emb.data.numpy().flatten()
    t_flat = t_emb.data.numpy().flatten()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.quiver(0, 0, 0, h_flat[0], h_flat[1], h_flat[2], color='xkcd:blue')
    ax.quiver(h_flat[0], h_flat[1], h_flat[2], r_flat[0], r_flat[1], r_flat[2], color='xkcd:green')
    ax.quiver(0, 0, 0, t_flat[0], t_flat[1], t_flat[2], color='xkcd:red')

    ax.text(h_flat[0] / 2, h_flat[1] / 2, h_flat[2] / 2, id2Entity[h])
    ax.text(h_flat[0] + (r_flat[0] / 2), h_flat[1] + (r_flat[1] / 2), h_flat[2] + (r_flat[2] / 2), id2Relation[r])
    ax.text(t_flat[0] / 2, t_flat[1] / 2, t_flat[2] / 2, id2Entity[t])

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()


def train(transx, batchSize, train_h, train_t, train_r, epochs):
    model = NegativeSampling(
        model=transx,
        loss=MarginLoss(margin=2.0),
        batch_size=batchSize)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        weight_decay=0)

    h_set, t_set = set(train_h), set(train_t)

    for epoch in range(epochs):
        batch_h = np.array(train_h + [-1] * batchSize + [-1] * batchSize, dtype=np.int64)
        batch_r = np.array(train_r + [-1] * batchSize + [-1] * batchSize, dtype=np.int64)
        batch_t = np.array(train_t + [-1] * batchSize + [-1] * batchSize, dtype=np.int64)
        batch_y = np.array([1] * batchSize + [-1] * batchSize + [-1] * batchSize, dtype=np.float32)

        # Let's corrupt head and tail
        for j in range(2):
            for i in range(batchSize):
                h, r, t = batch_h[i], batch_r[i], batch_t[i]

                while True:
                    if j == 0:
                        hp = random.sample(h_set, 1)[0]
                    else:
                        hp = h
                    if j == 1:
                        tp = random.sample(t_set, 1)[0]
                    else:
                        tp = t
                    if not present(train_h, train_r, train_t, hp, r, tp, missing=True):
                        break

                batch_h[i + (j + 1) * batchSize] = hp
                batch_r[i + (j + 1) * batchSize] = r
                batch_t[i + (j + 1) * batchSize] = tp

        optimizer.zero_grad()
        loss = model({
            'batch_h': to_var(batch_h),
            'batch_t': to_var(batch_t),
            'batch_r': to_var(batch_r),
            'batch_y': to_var(batch_y),
            'mode': 'normal'
        })
        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())
        # This loss works well for the example; completely empirical.
        if loss.item() < .8:
            break

def to_var(x):
    return Variable(torch.from_numpy(x))

def present(train_h, train_r, train_t, hp, r, tp, missing=False):
    found = False
    # Missing triples
    found = missing and (found or (hp == 1 and r == 100 and tp == 5))
    found = missing and (found or (hp == 12 and r == 101 and tp == 2))
    for i in range(len(train_h)):
        if found:
            break
        found = found or (hp == train_h[i] and r == train_r[i] and tp == train_t[i])
    return found

if __name__ == '__main__':
    plotVectors()