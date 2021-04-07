import pandas as pd

def getRulesByParameters(fileName, parameter = None, value = None):
    f = open(fileName)
    headers = ["Rule", "Head Coverage", "Std Confidence", "PCA Confidence", "Positive Examples", "Body size", "PCA Body size", "Functional variable"]
    i = 0
    rules = []
    for line in f:
        if i == 0:
            i+=1
            continue
        
        vals = line.strip().split(" ")
        newvals = []
        for val in vals:
            if val != '':
                newvals.append(val)
        vals = newvals
        newvals = []
        st = ""
        for i in range(len(vals)):
            if i<=3:
                st+=vals[i] + " "
            elif i == 4:
                newvals.append(st[:-1])
            
            if i >3:
                try:
                    newvals.append(float(vals[i]))
                except:
                    newvals.append(vals[i])
        rules.append(newvals)
    f.close()
    data = pd.DataFrame(rules, columns=headers)

    if parameter == None:
        return data
    else:
        return data[data[parameter] >= value]

rules = getRulesByParameters(r"C:\Users\nari9\OneDrive\Documents\PG\Assignments\Thesis\KGE-Combined\Rules\transe_0_m2_p0.7.txt", "Body size", 10000)

print (rules)



    