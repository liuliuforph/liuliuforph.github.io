import torch
import numpy as np
import torch.nn as nn
import time

def mom_obj(gen, A, y, zf, zg, batch_size, se_unreduced):
    # Shuffle the index
    shuffled_idx = torch.randperm(A.size()[0])
    
    A_shuffled = A[shuffled_idx, :]
    y_shuffled = y[shuffled_idx, :]
    
    # compute AG(zf), AG(zg)
    Af = A_shuffled.mm(gen(zf).view(-1,1))
    Ag = A_shuffled.mm(gen(zg).view(-1,1))
    
    # find (y_i - a_iG(zf))^2 , (y_i - a_iG(zg))^2 
    loss_1 = se_unreduced(Af, y_shuffled)
    loss_2 = se_unreduced(Ag, y_shuffled)

    # now find median block of loss_1 - loss_2
    loss_3 = loss_1 - loss_2
    loss_3 = loss_3[:batch_size*(A_shuffled.shape[0]//batch_size)] # make the number of rows a multiple of batch size
    loss_3 = loss_3.view(-1,batch_size) # reshape
    loss_3 = loss_3.mean(axis=1) # find mean on each batch
    loss_3_numpy = loss_3.detach().cpu().numpy() # convert to numpy

#     print(loss_3_numpy)
    
#     median_idx = np.argsort(loss_3_numpy)[loss_3_numpy.shape[0]//2] # sort and pick middle element

    C = 5
    
    median_idx = np.argsort(loss_3_numpy)[loss_3_numpy.shape[0]//2-C:loss_3_numpy.shape[0]//2+C] # sort and pick middle element

    select =[]
    
    for idx in median_idx:
        new_select = [ii for ii in range(idx*batch_size, batch_size*(idx+1))]
        select += new_select
        
        
    # pick median block
    loss_1_mom = loss_1[select,:]
    loss_2_mom = loss_2[select,:]
    
    loss_f = torch.mean(loss_1_mom - loss_2_mom)
    
    return loss_f



def MOM_estimate(gen, z0, A, y, batch_size, N_ITER, LR):
    device = 'cuda:0'

    # variable for f
    z_output = torch.zeros(z0.shape, device=device)
    
    zf = 1*torch.ones(z0.shape, device=device)
    zf.requires_grad = True

    # variable for g
#     zg = torch.ones(z0.shape, device=device)
    zg = -1*torch.ones(z0.shape, device=device)
    zg.requires_grad = True

    # learning rate, number of iterations and batch_size
    LRF = LR
    LRG = LR

    # array for storing error values
    loss_f_record = []
    loss_g_record = []


    # DO NOT REDUCE(via mean or sum) the square error terms
    # the corrupted samples become mixed with the
    # clean samples
    se_unreduced = nn.MSELoss(reduction='none')

    # adam optimizers
    opt_f = torch.optim.Adam([zf], lr=LRF)
    opt_g = torch.optim.Adam([zg], lr=LRG)

    x = gen(z0)

    count = 0
    n_features = x.view(-1).size()[0]

    
    time_elapsed = []
    time_start = time.clock()
    
    for i in range(N_ITER):
        # take descen step in zf. 
        # Backward on a loss function; then use a step for a variable.
        if i % 500 == 0:
            print(i)
            
        opt_f.zero_grad()
        opt_g.zero_grad()
        
        loss_f = mom_obj(gen, A, y, zf, zg, batch_size, se_unreduced)

        
        loss_f.backward(retain_graph=True)
        opt_f.step()                                  
    
#         loss_g.backward()
        (-loss_f).backward()
        opt_g.step()
    

        # record error value
        loss_f_record.append(torch.norm(gen(zf).detach()-x).item()**2/n_features)
        loss_g_record.append(torch.norm(gen(zg).detach()-x).item()**2/n_features)

        time_elapsed.append(time.clock() - time_start)

        if i >= N_ITER - 200:
            z_output = z_output + zf
            count += 1
    
    return loss_f_record, loss_g_record, z_output/count, zg, time_elapsed
