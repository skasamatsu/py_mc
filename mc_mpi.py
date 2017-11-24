import os,sys
import numpy as np
from mpi4py import MPI
import pickle

from mc import *

class ParallelMC(object):
    def __init__(self, comm, MCalgo, model, configs, kTs, grid=None, writefunc=write_energy, subdirs=True):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.procs = self.comm.Get_size()
        self.kTs = kTs
        self.model = model
        self.subdirs = subdirs
        self.nreplicas = len(configs)

        if not(self.procs == self.nreplicas == len(self.kTs)):
            if self.rank==0:
                print("ERROR: You have to set the number of replicas equal to the"
                      +"number of temperatures equal to the number of processes"
                )
            sys.exit(1)

        myconfig = configs[self.rank]
        mytemp = kTs[self.rank]
        self.mycalc = MCalgo(model, mytemp, myconfig, writefunc, grid)
        
    def run(self, nsteps, sample_frequency, observefunc=lambda *args: None):
        if self.subdirs:
            # make working directory for this rank
            try:
                os.mkdir(str(self.rank))
            except FileExistsError:
                pass
            os.chdir(str(self.rank))
        observables = self.mycalc.run(nsteps, sample_frequency, observefunc)
        pickle.dump(self.mycalc.config, open("config.pickle","wb"))
        if self.subdirs: os.chdir("../")
        if sample_frequency:
            obs_buffer = np.empty([self.procs,len(observables)])
            self.comm.Allgather(observables, obs_buffer)
            return obs_buffer

        
class TemperatureRX_MPI(ParallelMC):
    def __init__(self, comm, MCalgo, model, configs, kTs, swap_algo=swap_configs, writefunc=write_energy_Temp, subdirs=True):
        super(TemperatureRX_MPI, self).__init__(comm, MCalgo, model, configs, kTs, writefunc, subdirs)
        self.swap_algo = swap_algo
        self.betas = 1.0/np.array(kTs)
        self.energyRankMap = np.zeros(len(kTs))
        #self.energyRankMap[i] holds the energy of the ith rank
        self.T_to_rank = np.arange(0, self.procs, 1)
        # self.T_to_rank[i] holds the rank of the ith temperature

    def reload(self):
        self.T_to_rank = pickle.load(open("T_to_rank.pickle","rb"))
        self.mycalc.kT = self.kTs[self.myTindex()]
        self.mycalc.config = pickle.load(open(str(self.rank)+"/calc.pickle","rb"))
        
    def myTindex(self):
        for i in range(self.nreplicas):
            if self.T_to_rank[i] == self.rank:
                return i
        os.exit("Internal error in TemperatureRX_MPI.rank_to_T")


    def Xtrial(self, XCscheme=-1):
        # Gather energy to root node
        #print(type(self.mycalc.energy))
        self.comm.Gather([self.mycalc.energy, MPI.DOUBLE], self.energyRankMap, root=0)
        if self.rank == 0:
            if XCscheme == 0 or XCscheme == 1:
                # exchanges between 0-1, 2-3, 4-5, ... or 1-2, 3-4, 5-6 are tried
                # 0 corresponds to lowest energy replica and resides in rank self.T_to_rank[0]
                trialreplica = XCscheme
                while trialreplica + 1 < self.procs:
                    rank_low = self.T_to_rank[trialreplica]
                    rank_high = self.T_to_rank[trialreplica + 1]
                    delta = (self.betas[trialreplica+1] - self.betas[trialreplica]) \
                    *(self.energyRankMap[rank_low] - self.energyRankMap[rank_high])

                    if delta < 0.0:
                        tmp = self.T_to_rank[trialreplica]
                        self.T_to_rank[trialreplica] = self.T_to_rank[trialreplica+1]
                        self.T_to_rank[trialreplica+1] = tmp
                        #print("RXtrial accepted")
                    else:
                        accept_probability = exp(-delta)
                        #print accept_probability, "accept prob"
                        if random() <= accept_probability:
                            tmp = self.T_to_rank[trialreplica]
                            self.T_to_rank[trialreplica] = self.T_to_rank[trialreplica+1]
                            self.T_to_rank[trialreplica+1] = tmp
                            #print("RXtrial accepted")
                        else:
                            #print("RXtrial rejected")
                            pass
                    trialreplica += 2
        self.comm.Bcast(self.T_to_rank, root=0)
        #print(self.T_to_rank)
        #sys.exit()
        self.mycalc.kT = self.kTs[self.myTindex()]

    
    def run(self, nsteps, RXtrial_frequency, sample_frequency,
            observfunc=lambda *args: None, subdirs=True):
        if subdirs:
            try:
                os.mkdir(str(self.rank))
            except FileExistsError:
                pass     
            os.chdir(str(self.rank))
        self.accept_count = 0
        self.mycalc.energy = self.mycalc.model.energy(self.mycalc.config)
        if hasattr(observfunc(self.mycalc,open(os.devnull,"w")),"__iter__"):
            obs_len = len(observfunc(self.mycalc,open(os.devnull,"w")))
            obs = np.zeros([len(self.kTs), obs_len])
        nsample = 0
        XCscheme = 0
        output = open("obs.dat", "a")
        if not sample_frequency:
            sample_frequency = float("inf")
        for i in range(nsteps):
            self.mycalc.MCstep()
            sys.stdout.flush()
            if i%RXtrial_frequency == 0:
                self.Xtrial(XCscheme)
                XCscheme = (XCscheme+1)%2
            if i%sample_frequency == 0:
                obs[self.myTindex()] += observfunc(self.mycalc, output)
                nsample += 1
        
        pickle.dump(self.mycalc.config, open("calc.pickle","wb"))
        
        if subdirs: os.chdir("../")

        if self.rank == 0:
            pickle.dump(self.T_to_rank, open("T_to_rank.pickle","wb"))

        if nsample != 0:
            obs = np.array(obs)
            obs_buffer = np.empty(obs.shape)
            obs /= nsample
            self.comm.Allreduce(obs, obs_buffer, op=MPI.SUM)
            return obs_buffer
        
        
