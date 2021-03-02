from DataLoader.TripleManager import TripleManager
from .ModelUtils import ModelUtils
import sys
import os
import pickle
import glob
import torch

def check():
    #folder = sys.argv[1]

    folder = ""

    all_model_files=[]
    for trial_file in glob.glob(folder + "Ax/*.trial"):
        filename, ext = os.path.splitext(os.path.basename(trial_file))

        splitname = filename.split('_')

        model_name = splitname[0]
        dataset = int(splitname[1])
        trial_index = int(splitname[2])

        dataset_name = ""
        if dataset == 0:
            dataset_name = "FB13"
        if dataset == 1:
            dataset_name = "FB15K"
        if dataset == 2:
            dataset_name = "FB15K237"
        if dataset == 3:
            dataset_name = "NELL-995"
        if dataset == 4:
            dataset_name = "WN11"
        if dataset == 5:
            dataset_name = "WN18"
        if dataset == 6:
            dataset_name = "WN18RR"
        if dataset == 7:
            dataset_name = "YAGO3-10"

        print("Model:", model_name, "; Dataset:", dataset_name, "; Trial: ", trial_index)

        manager = TripleManager(folder + "Datasets/" + dataset_name + "/", splits=["new_test"],
                                corruption_mode='Global')

        # Get parameters and index for trial
        with open(trial_file, 'rb') as f:
            parameters = pickle.load(f)

        mu = ModelUtils(model_name, parameters)
        model_file = folder + "Model/" + str(dataset) + "/" + mu.get_name() + ".model"

        if not os.path.exists(model_file):
            if os.path.exists(folder + "Ax/" + filename + ".result"):
                print('Model is not present: ', model_file)
            elif not os.path.exists(folder + "Ax/" + filename + ".fail") and not os.path.exists(folder + "Ax/" + filename + ".result"):
                print('Model is pending: ', model_file)
        else:
            all_model_files.append(os.path.basename(model_file))
            # Load model and check it has no NaNs.
            model = mu.get_model(manager.entityTotal, manager.relationTotal, 0)
            model.model.load_checkpoint(model_file)

            contains_nan = False
            for p in model.parameters():
                if True in torch.isnan(p.data):
                    contains_nan = True
                    break

            # Move to NaN!
            if contains_nan:
                print('Model: ', model_file, ' contains NaNs! Trial: ', parameters['trial_index'], '; Fail file exists: ',
                      os.path.exists(folder + "Ax/" + model_name + "_" + str(dataset) + "_" + str(parameters['trial_index']) + ".fail"))
                print('mv ' + model_file.replace('.model', '.*') + ' ' +
                      model_file.replace(os.path.basename(model_file), '').replace('Model/', 'Model/NaN/'))

    # For the rest of the models that are not coming from the trials, remove!
    for model_file in glob.glob(folder + "Model/*/*.model"):
        if os.path.basename(model_file) not in all_model_files:
            print('mv ' + model_file.replace('.model', '.*') + ' ' + model_file.replace(os.path.basename(model_file), '').replace('Model/', 'Model/NotFound/'))

if __name__ == '__main__':
    check()