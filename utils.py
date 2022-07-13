import random
import os
import numpy as np
import torch
from torch.autograd import Variable
import sklearn.preprocessing as pp
import sklearn.metrics.pairwise as pw
from sklearn.metrics.pairwise import rbf_kernel


def seed_setting(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def cos_similarity(x1,x2):
    t1 = x1.dot(x2.T)

    x1_linalg = np.linalg.norm(x1,axis=1)
    x2_linalg = np.linalg.norm(x2,axis=1)
    x1_linalg = x1_linalg.reshape((x1_linalg.shape[0],1))
    x2_linalg = x2_linalg.reshape((1,x2_linalg.shape[0]))
    t2 = x1_linalg.dot(x2_linalg)
    cos = t1/t2

    return cos

def calculate_S(z):
    S = pw.cosine_similarity(z, z)
    return S

def calculate_S_multi(z1, z2):  # z: np.ndarray
    #z = torch.nn.functional.normalize(z, p=2, dim=1)
    #S = torch.matmul(z1, z2.T)
    z1 = pp.normalize(z1, norm='l2')
    z2 = pp.normalize(z2, norm='l2')
    S = np.matmul(z1, z2.T)
    return S

def calculate_rou(S, rate=0.4):
    m = S.shape[0]
    rou = np.zeros((m))

    t = int(rate * m * m)
    temp = np.sort(S.reshape((m * m,)))
    Sc = temp[-t]
    rou = np.sum(np.sign(S - Sc), axis=1) - np.sign(S.diagonal() - Sc)

    return rou

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

def CalcSim(label1, label2):
    # calculate the similar matrix
    #if use_gpu:
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    #else:
    #    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim

def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))
    out_affnty = affnty/col_sum # row data sum = 1
    in_affnty = np.transpose(affnty/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty # col, row

def zero2eps(x):
    x[x == 0] = 1
    return x