from TripleManager import TripleManager
from TransE import TransE
from TransEManual import TransEManual
from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss
from Trainer import Trainer
import time
import sys

if __name__ == '__main__':
    #folder = sys.argv[1]
    #dataset = int((int(sys.argv[2])-1)/8)
    #negative_ent = 2**((int(sys.argv[2])-1)-(dataset*8))
    #corruptionMode = sys.argv[3]
    #modelName = sys.argv[4]
    #p_norm = int(sys.argv[5])
    #margin = float(sys.argv[6])

    folder = ""
    dataset = 0
    negative_ent = 2 ** 1
    corruptionMode = "LCWA"
    modelName = "manual_ours"
    p_norm = 2
    margin = 0.01

    dataset_name = ""
    if dataset==0:
        dataset_name="FB13"
    if dataset==1:
        dataset_name="FB15K"
    if dataset==2:
        dataset_name="FB15K237"
    if dataset==3:
        dataset_name="NELL-995"
    if dataset==4:
        dataset_name="WN11"
    if dataset==5:
        dataset_name="WN18"
    if dataset==6:
        dataset_name="WN18RR"
    if dataset==7:
        dataset_name="YAGO3-10"

    print("Dataset: " + dataset_name + "; NegRate: " + str(negative_ent) + \
          "; Mode: " + corruptionMode + "; Norm: " + str(p_norm))

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    nbatches = 100
    negative_rel = 0
    seed = 0
    train_dataloader = TripleManager(path, splits=["train"], nbatches=nbatches, neg_ent=negative_ent,
                                     neg_rel=negative_rel, seed=seed, corruptionMode=corruptionMode)

    model = None

    norm_flag = True
    normalize_r = True
    norm_start = False
    alpha=0.01

    if modelName.endswith("notr"):
        normalize_r = False
    if modelName.endswith("begin"):
        norm_flag = False
        norm_start = True

    if modelName.startswith("openke"):
        transx = TransE(
            ent_tot=train_dataloader.entityTotal,
            rel_tot=train_dataloader.relationTotal,
            dim=200,
            p_norm=p_norm,
            norm_flag=norm_flag,
            normalize_r=normalize_r,
            norm_start=norm_start,
            seed=seed)

        model = NegativeSampling(
            model=transx,
            loss=MarginLoss(margin=margin),
            batch_size=train_dataloader.batchSize)
    elif modelName.startswith("manual"):
        mode = None
        if "pge" in modelName:
            mode = "pge"
        elif "ours" in modelName:
            mode = "ours"

        model = TransEManual(
            ent_tot=train_dataloader.entityTotal,
            rel_tot=train_dataloader.relationTotal,
            dim=200,
            alpha=alpha,
            mode=mode,
            norm_flag=norm_flag,
            normalize_r=normalize_r,
            norm_start=norm_start,
            seed=seed,
            batch_size=train_dataloader.batchSize)
    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000,
            alpha=alpha, use_gpu=False, save_steps=50, early_stopping_enabled=False,
            checkpoint_dir=folder + "Model/" + modelName + "_" + dataset_name + "_" + str(negative_ent) + "_" + \
                           corruptionMode + "_" + str(p_norm))
    trainer.run()
    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))