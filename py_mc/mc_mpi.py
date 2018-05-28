import os,sys
import numpy as np
from mpi4py import MPI
import pickle
import random as rand

from py_mc.mc import *

def RX_MPI_init():
    args = sys.argv
    nreplicas = int(args[1] )
    nprocs_per_replica = int(args[2])
    commworld = MPI.COMM_WORLD
    worldrank = commworld.Get_rank()
    worldprocs = commworld.Get_size()
    rand_seeds = [rand.random() for i in range(worldprocs)]
    rand.seed(rand_seeds[worldrank])
    
    if worldprocs > nreplicas:
        if worldrank == 0:
            print("Setting number of replicas smaller than MPI processes; I hope you"
                  +" know what you're doing..."
            )
            sys.stdout.flush()
        if worldrank >= nreplicas:
            # belong to comm that does nothing
            comm = commworld.Split(color=1, key=worldrank)
            comm.Free()
            sys.exit() # Wait for MPI_finalize
        else:
            comm = commworld.Split(color=0, key=worldrank)
    else:
        comm = commworld
    return comm, nreplicas, nprocs_per_replica


class ParallelMC(object):
    def __init__(self, comm, MCalgo, model, configs, kTs, grid=None, subdirs=True):
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
        self.mycalc = MCalgo(model, mytemp, myconfig, grid)
        
    def run(self, nsteps, sample_frequency, observer=observer_base()):
        if self.subdirs:
            # make working directory for this rank
            try:
                os.mkdir(str(self.rank))
            except FileExistsError:
                pass
            os.chdir(str(self.rank))
        observables = self.mycalc.run(nsteps, sample_frequency, observer)
        pickle.dump(self.mycalc.config, open("config.pickle","wb"))
        if self.subdirs: os.chdir("../")
        if sample_frequency:
            obs_buffer = np.empty([self.procs,len(observables)])
            self.comm.Allgather(observables, obs_buffer)
            return obs_buffer

        
class TemperatureRX_MPI(ParallelMC):
    def __init__(self, comm, MCalgo, model, configs, kTs, grid=None,  subdirs=True):
        super(TemperatureRX_MPI, self).__init__(comm, MCalgo, model, configs, kTs, grid, subdirs)
        self.betas = 1.0/np.array(kTs)
        self.energyRankMap = np.zeros(len(kTs))
        #self.energyRankMap[i] holds the energy of the ith rank
        self.T_to_rank = np.arange(0, self.procs, 1, dtype=np.int)
        # self.T_to_rank[i] holds the rank of the ith temperature
        self.rank_to_T = np.arange(0, self.procs, 1, dtype=np.int)
        self.float_buffer = np.array(0.0, dtype=np.float)
        self.int_buffer = np.array(0, dtype=np.int)
        self.obs_save = []
        self.Trank_hist = []
        self.kT_hist = []

    def reload(self):
        self.rank_to_T = pickle.load(open("rank_to_T.pickle","rb"))
        self.mycalc.kT = self.kTs[self.myTindex()]
        self.mycalc.config = pickle.load(open(str(self.rank)+"/calc.pickle","rb"))
        self.obs_save0 = np.load(open(str(self.rank)+"/obs_save.npy","rb"))
        self.Trank_hist0 = np.load(open(str(self.rank)+"/Trank_hist.npy","rb"))
        self.kT_hist0 = np.load(open(str(self.rank)+"/kT_hist.npy","rb"))
        
    def myTindex(self):
        for i in range(self.nreplicas):
            if self.T_to_rank[i] == self.rank:
                return i
        sys.exit("Internal error in TemperatureRX_MPI.rank_to_T")

    def find_procrank_from_Trank(self, Trank):
        i = np.argwhere(self.rank_to_T == Trank)
        if i == None:
            sys.exit("Internal error in TemperatureRX_MPI.find_procrank_from_Trank")
        else:
            return i

    def Xtrial(self, XCscheme=-1):
        # What is my temperature rank?
        myTrank = self.rank_to_T[self.rank]
        if (myTrank+XCscheme)%2 == 0 and myTrank == self.procs - 1:
            self.comm.Allgather(self.rank_to_T[self.rank], self.rank_to_T)
            return
        if XCscheme == 1 and myTrank == 0:
            self.comm.Allgather(self.rank_to_T[self.rank], self.rank_to_T)
            return
        if (myTrank+XCscheme)%2 == 0:
            myTrankp1 = myTrank + 1
            # Get the energy from the replica with higher temperature
            exchange_rank = self.find_procrank_from_Trank(myTrankp1)
            self.comm.Recv(self.float_buffer, source=exchange_rank, tag=1)
            delta = (self.betas[myTrankp1] - self.betas[myTrank]) \
                    * (self.mycalc.energy - self.float_buffer)
            if delta < 0.0:
                # Exchange temperatures!
                self.comm.Send([self.rank_to_T[self.rank],1,MPI.INT], dest=exchange_rank, tag=2)
                
                self.rank_to_T[self.rank] = myTrankp1
            else:
                accept_probability = exp(-delta)
                #print accept_probability, "accept prob"
                if random() <= accept_probability:
                    self.comm.Send([self.rank_to_T[self.rank],1,MPI.INT], dest=exchange_rank, tag=2)
                    self.rank_to_T[self.rank] = myTrankp1
                else:
                    #print("RXtrial rejected")
                    self.comm.Send([self.rank_to_T[exchange_rank],1,MPI.INT], dest=exchange_rank, tag=2)
        else:
            myTrankm1 = myTrank - 1
            exchange_rank = self.find_procrank_from_Trank(myTrankm1)
            self.comm.Send(self.mycalc.energy, dest=exchange_rank, tag=1)
            self.comm.Recv([self.int_buffer,1,MPI.INT],source=exchange_rank, tag=2)
            self.rank_to_T[self.rank] = self.int_buffer
        self.comm.Allgather(self.rank_to_T[self.rank], self.rank_to_T)
        self.mycalc.kT = self.kTs[self.rank_to_T[self.rank]]
        return

        '''
        # Gather energy to root node
        #print(type(self.mycalc.energy))
        self.comm.Gather([np.float64(self.mycalc.energy), MPI.DOUBLE], self.energyRankMap, root=0)
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
        '''
    
    def run(self, nsteps, RXtrial_frequency, sample_frequency=verylargeint,
            print_frequency=verylargeint,
            observer=observer_base(), subdirs=True, save_obs=True):
        if subdirs:
            try:
                os.mkdir(str(self.rank))
            except FileExistsError:
                pass     
            os.chdir(str(self.rank))
        self.accept_count = 0
        self.mycalc.energy = self.mycalc.model.energy(self.mycalc.config)
        if hasattr(observer.observe(self.mycalc,open(os.devnull,"w")),"__iter__"):
            obs_len = len(observer.observe(self.mycalc,open(os.devnull,"w")))
            obs = np.zeros([len(self.kTs), obs_len])
        if hasattr(observer.observe(self.mycalc,open(os.devnull,"w")),'__add__'):
            observe = True
        else:
            observe = False
        nsample = 0
        XCscheme = 0
        output = open("obs.dat", "a")
        for i in range(1,nsteps+1):
            self.mycalc.MCstep()
            if i%RXtrial_frequency == 0:
                self.Xtrial(XCscheme)
                XCscheme = (XCscheme+1)%2
            if i%sample_frequency == 0 and observe:
                obs_step = observer.observe(self.mycalc, output,i%print_frequency==0)
                obs[self.rank_to_T[self.rank]] += obs_step
                if save_obs:
                    self.obs_save.append(obs_step)
                    self.Trank_hist.append(self.rank_to_T[self.rank])
                    self.kT_hist.append(self.mycalc.kT)
                nsample += 1
        
        pickle.dump(self.mycalc.config, open("calc.pickle","wb"))
        if save_obs:
            if hasattr(self, "obs_save0"):
                obs_save_ = np.concatenate((self.obs_save0, np.array(self.obs_save)))
                Trank_hist_ = np.concatenate((self.Trank_hist0, np.array(self.Trank_hist)))
                kT_hist_ = np.concatenate((self.kT_hist0, np.array(self.kT_hist)))
            else:
                obs_save_ = np.array(self.obs_save)
                Trank_hist_ = np.array(self.Trank_hist)
                kT_hist_ = np.array(self.kT_hist)
            
            np.save(open("obs_save.npy","wb"), obs_save_, False)
            np.save(open("Trank_hist.npy", "wb"), Trank_hist_, False)
            np.save(open("kT_hist.npy", "wb"), kT_hist_, False)
        
        if subdirs: os.chdir("../")

        if self.rank == 0:
            pickle.dump(self.rank_to_T, open("rank_to_T.pickle","wb"))
            np.save(open("kTs.npy", "wb"), self.kTs, False)

        if nsample != 0:
            obs = np.array(obs)
            obs_buffer = np.empty(obs.shape)
            obs /= nsample
            self.comm.Allreduce(obs, obs_buffer, op=MPI.SUM)
            obs_list = []
            args_info = observer.obs_info(self.mycalc)
            for i in range(len(self.kTs)):
                obs_list.append(obs_decode(args_info,obs_buffer[i]))
            return obs_list
        
        
