import torch
import numpy as np
import torch.nn as nn
from math import floor
import time

def trim_obj(gen, A, y, zf, se_unreduced):

    Af = A.mm(gen(zf).view(-1,1))
    
    loss_1 = se_unreduced(Af, y)


    loss_1_numpy = loss_1.detach().cpu().numpy().reshape(-1) # convert to numpy
    
    smallest_idx = np.argsort(loss_1_numpy)[:floor(loss_1_numpy.shape[0]*.8)] # 
    
    loss_1_trim = loss_1[smallest_idx]
#     print(loss_1_trim.shape)
    loss_f = torch.mean(loss_1_trim)
    
    return loss_f



def trim_estimate(gen, z0, A, y, N_ITER, LR):
    device = 'cuda:0'


    # variable for f
    zf = torch.ones(z0.shape, device=device)
    zf.requires_grad = True



    # learning rate, number of iterations and batch_size
    LRF = LR

    # array for storing error values
    loss_trim = []


    # DO NOT REDUCE(via mean or sum) the square error terms
    # the corrupted samples become mixed with the
    # clean samples
    se_unreduced = nn.MSELoss(reduction='none')

    # adam optimizers
    opt_f = torch.optim.Adam([zf], lr=LRF)


    # # SGD
    # opt_f = torch.optim.SGD([zf], lr=LRF)
    # opt_g = torch.optim.SGD([zg], lr=LRG)

#     output_zf = zf
#     output_zg = zg

    x = gen(z0)

    count = 0
    n_features = x.view(-1).size()[0]
    
    time_elapsed = []
    time_start = time.clock()

    for i in range(N_ITER):
        # take descen step in zf. 
        # Backward on a loss function; then use a step for a variable.
        if i % 1000 == 0:
            print(i)
        opt_f.zero_grad()
        loss_f = trim_obj(gen, A, y, zf, se_unreduced)
        loss_f.backward()
        opt_f.step()  

        # record error value
        loss_trim.append(torch.norm(gen(zf).detach()-x).item()**2/n_features)
        
        time_elapsed.append(time.clock() - time_start)

    return loss_trim, zf, time_elapsed
