import numpy as np
from mpi4py import MPI

def jk(input):
    return np.ones(500)
def objfunc(input):
    comm = MPI.COMM_WORLD
    cores = comm.Get_size()
    rank = comm.Get_rank()
    split = int(len(input)/cores)
    if len(input)%cores ==0:
        input = input[2*rank:2*rank+split]
    else:
        inds = split*np.arange(1,cores+1)
        einds = np.concatenate((np.arange(1,len(input)-cores*split+1),(len(input)-cores*split+1)*np.ones((cores-(len(input)%cores)))))
        inds = inds+einds
        start_inds = np.insert(inds,0,0)
        input = input[int(start_inds[rank]):int(inds[rank])]
        output =[]
    for i in range(len(input)):
        output.append(jk(input[i]))
    massout =None
    if rank==0:
        massout = []*cores
    count = [inds[i]-inds[i-1] for i in range(1,len(inds))]
    #comm.Gatherv(output,[massout,count,start_inds],root=0)
    massout = comm.gather(output,root=0)
    if rank==0:
        actout =[]
        actout = [item for sublist in massout for item in sublist]
        return actout


