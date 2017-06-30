import os,sys
from mpi4py import MPI
import pickle

from mc import *

class ParallelMC:
    def __init__(self, MCalgo, model, configs, kTs, writefunc=write_energy, subdirs=True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.procs = self.comm.Get_size()
        self.kTs = kTs
        self.nreplicas = len(configs)
        self.model = model
        self.subdirs = subdirs

        if not(self.procs == self.nreplicas == len(self.kTs)):
            if self.rank==0:
                print("ERROR: You have to set the number of replicas equal to the "
                "number of processes equal to the number of temperatures"
                )
            sys.exit(1)

        myconfig = configs[self.rank]
        mytemp = kTs[self.rank]
        self.mycalc = MCalgo(model, mytemp, myconfig, writefunc)
        
    def run(self, nsteps, sample_frequency):
        if self.subdirs:
            # make working directory for this rank
            try:
                os.mkdir(str(self.rank))
            except FileExistsError:
                pass
            os.chdir(str(self.rank))
        self.mycalc.run(nsteps, sample_frequency)
        pickle.dump(self.mycalc.config, open("config.pickle","wb"))
        if self.subdirs: os.chdir("../")
        
        
            

