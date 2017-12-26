from math import exp
from random import random,randrange
from multiprocessing import Process, Queue, Pool, TimeoutError
import os, sys
import numpy as np

'''Defines base classes for Monte Carlo simulations'''


class model:
    ''' This class defines a model whose energy equals 0 no matter the configuration, and the configuration
    never changes.
    This is a base template for building useful models.'''
    
    model_name = None
    
    #def __init__(self):
        

    def energy(self, config):
        '''Calculate energy of configuration: input: config'''
        return 0.0

    def trialstep(self, config, energy):
        '''Define a trial step on config. Returns dconfig, which can contain the minimal information for
        constructing the trial configuration from config to be used in newconfig(). Make sure that
        config is the same upon entry and exit'''
        dE = 0.0
        # Return only change in configuration dconfig so that
        # you don't have to copy entire configurations,
        # which can sometimes be costly
        return dconfig, dE

    def newconfig(self, config, dconfig):
        '''Build new configuration from config and dconfig.'''
        return config

def write_energy(MCcalc):
    with open("energy.out", "a") as f:
        f.write(str(MCcalc.energy)+"\n")
        f.flush()

def write_energy_Temp(MCcalc, outputfile=open("energy.out", "a")):
    outputfile.write(str(MCcalc.energy)+"\t"+str(MCcalc.kT)+"\n")
    outputfile.flush()

class grid_1D:
    def __init__(self, dx, minx, maxx):
        self.dx = dx
        self.x = np.arange(minx, maxx, dx)
    
class CanonicalMonteCarlo:

    def __init__(self, model, kT, config, writefunc=write_energy,
                 grid=0):
        self.model = model
        self.config = config
        self.kT = kT
        #self.energy = self.model.energy(self.config)
        self.writefunc = writefunc
        self.grid = grid

    def MCstep(self):
        dconfig, dE  = self.model.trialstep(self.config, self.energy)
        if dE < 0.0:
            self.config = self.model.newconfig(self.config, dconfig)
            self.energy += dE
            #print "trial accepted"
        else:
            accept_probability = exp(-dE/self.kT)
            if random() <= accept_probability:
                self.config = self.model.newconfig(self.config, dconfig)
                self.energy += dE
                #print "trial accepted"

    def run(self, nsteps, sample_frequency=0, observefunc=lambda *args: None):
        if not sample_frequency:
            sample_frequency = float("inf")
            
        observables = 0.0
        nsample = 0
        self.energy = self.model.energy(self.config)
        output = open("obs.dat", "a")
        if hasattr(observefunc(self,output),'__add__'):
            observe = True
        else:
            observe = False
            
        for i in range(nsteps):
            self.MCstep()
            sys.stdout.flush()
            if i%sample_frequency == 0 and observe:
                #self.writefunc(self)
                observables += observefunc(self,output)
                nsample += 1
        if nsample > 0:
            return observables/nsample
        else:
            return None
        

                
def MCalgo_Run_multiprocess_wrapper(MCcalc, nsteps, sample_frequency=0, outdir=None):
    if outdir:
        print("got into wrapper")
        # create subdirectory and run there
        cwd = os.getcwd()
        if not os.path.exists(outdir): os.mkdir(outdir)
        os.chdir(outdir)
        MCcalc.run(nsteps, sample_frequency)
        os.chdir(cwd)
    else:
        MCcalc.run(nsteps, sample_frequency)
    return MCcalc

def MultiProcessReplicaRun(MCcalc_list, nsteps, pool, sample_frequency=0, subdirs=False):
    n_replicas = len(MCcalc_list)
    if subdirs:
        print("subdirs")
        results = [
            pool.apply_async(
                MCalgo_Run_multiprocess_wrapper,(MCcalc_list[rep],
                                                 nsteps, sample_frequency, str(rep))
            )
            for rep in range(n_replicas)
        ]
        print("after apply_async")
    else:
        print("not subdirs")
        results = [
            pool.apply_async(
                MCalgo_Run_multiprocess_wrapper,(MCcalc_list[rep],
                                                 nsteps, sample_frequency)
            )
            for rep in range(n_replicas)
        ]
    results_list = [res.get(timeout=1800) for res in results]
    print("after res.get")
    for result in results:
        if not result.successful():
            sys.exit("Something went wrong")
    return results_list


def swap_configs(MCreplicas, rep, accept_count):
    # swap configs, energy
    tmp = MCreplicas[rep+1].config
    tmpe = MCreplicas[rep+1].energy
    MCreplicas[rep+1].config = MCreplicas[rep].config
    MCreplicas[rep+1].energy = MCreplicas[rep].energy
    MCreplicas[rep].config = tmp
    MCreplicas[rep].energy = tmpe
    accept_count += 1
    return MCreplicas, accept_count
    
            
class TemperatureReplicaExchange:

    def __init__(self, model, kTs, configs, MCalgo, swap_algo=swap_configs, writefunc=write_energy):
        assert len(kTs) == len(configs)
        self.model = model
        self.kTs = kTs
        self.betas = 1.0/kTs
        self.n_replicas = len(kTs)
        self.MCreplicas = []
        self.accept_count = 0
        self.swap_algo = swap_algo
        self.writefunc = writefunc
        self.configs = configs
        for i in range(self.n_replicas):
            self.MCreplicas.append(MCalgo(model, kTs[i], configs[i], writefunc))

    def Xtrial(self):
        # pick a replica
        rep = randrange(self.n_replicas - 1)
        delta = (self.betas[rep + 1] - self.betas[rep]) \
                *(self.MCreplicas[rep].energy -
                  self.MCreplicas[rep+1].energy)
        #print self.MCreplicas[rep].energy, self.model.energy(self.MCreplicas[rep].config)
                
        if delta < 0.0:
            self.MCreplicas, self.accept_count = self.swap_algo(self.MCreplicas,
                                                           rep, self.accept_count)
            print("RXtrial accepted")
        else:
            accept_probability = exp(-delta)
            #print accept_probability, "accept prob"
            if random() <= accept_probability:
                self.MCreplicas, self.accept_count = self.swap_algo(self.MCreplicas,
                                                           rep, self.accept_count)
                print("RXtrial accepted")
            else:
                print("RXtrial rejected")
        
    def run(self, nsteps, RXtrial_frequency, pool, sample_frequency=0, subdirs=False):
        self.accept_count = 0
        outerloop = nsteps//RXtrial_frequency
        for i in range(outerloop):
            self.MCreplicas = MultiProcessReplicaRun(self.MCreplicas, RXtrial_frequency, pool, sample_frequency, subdirs)
            self.Xtrial()
        self.configs = [MCreplica.config for MCreplica in self.MCreplicas]
        #print self.accept_count
        #self.accept_count = 0


def obs_encode(*args):
    nargs = np.array([len(args)])
    args_length_list = []
    obs_array = np.array([])
    for arg in args:
        obs_array = np.concatenate((obs_array, np.array(arg)))
        args_length_list.append(len(arg))
    args_length_array = np.array(args_length_list)
    args_info = np.concatenate((nargs, args_length_array))
    return  np.concatenate((args_info, obs_array))

def obs_decode(obs):
    nargs = np.around(obs[0]).astype(int)
    args_length_array = np.around(obs[1:nargs+1]).astype(int)
    args = []
    idx = nargs+1
    for i in range(nargs):
        length = args_length_array[i]
        args.append(obs[idx:idx+length])
        idx += length
    return args

