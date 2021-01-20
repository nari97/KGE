import os

class Loader:

    def __init__(self, folder):
        self.folder = folder
        #print (os.getcwd())
        entityFile = open(os.getcwd() + '/Datasets/' + self.folder +'/entity2id.txt')
        self.entities = {}
        

        i = 0
        for line in entityFile:
            if i==0:
                pass
            else:
                e = line.strip().split("\t")
                self.entities[e[0]] = e[1]
            i+=1
        entityFile.close()

        relationFile = open(os.getcwd() + '/Datasets/' + self.folder+'/relation2id.txt')
        self.relations = {}
        

        i = 0
        for line in relationFile:
            if i==0:
                pass
            else:
                e = line.strip().split(" ")
                if len(e)<2:
                    e = line.strip().split("\t")
                #print (e)
                self.relations[e[0]] = e[1]
            i+=1
        relationFile.close()
        
        dataFile = open(os.getcwd() + '/Datasets/' + self.folder +'/new_train2id.txt')
        self.dataset = []
        i = 0
        for line in dataFile:
            if i == 0:
                pass
            else:
                
                e = line.strip().split(" ")
                e[0] = int(e[0])
                e[1] = int(e[1])
                e[2] = int(e[2])
                self.dataset.append(e)
            i+=1
        dataFile.close()

        dataFile = open(os.getcwd() + '/Datasets/' + self.folder +'/new_valid2id.txt')
        self.validDataset = []
        i = 0
        for line in dataFile:
            if i == 0:
                pass
            else:
                e = line.strip().split(" ")
                e[0] = int(e[0])
                e[1] = int(e[1])
                e[2] = int(e[2])
                self.validDataset.append(e)
            i+=1
        dataFile.close()

        
    def getEntities(self):
        return self.entities

    def getRelations(self):
        return self.relations

    def getTrainDataset(self):  
        return self.dataset

    def getValidDataset(self):
        return self.validDataset

    def getEntityAndRelationCount(self):
        return len(self.entities), len(self.relations)



            


