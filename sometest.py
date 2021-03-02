import torch
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import wilcoxon

a = np.array([i for i in range(100)])
b = np.array([i for i in range(100,200)])

print (wilcoxon(a,b, zero_method = 'pratt'))