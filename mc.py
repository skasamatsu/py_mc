from math import exp
from random import random,randrange
from multiprocessing import Process, Queue, Pool, TimeoutError

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

    def trialstep(self, config):
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
        
class CanonicalMonteCarlo:

    def __init__(self, model, kT, config):
        self.model = model
        self.config = config
        self.energy = self.model.energy(self.config)
        self.kT = kT

    def MCstep(self):
        dconfig, dE  = self.model.trialstep(self.config)
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

    def run(self, nsteps):
        for i in range(nsteps):
            self.MCstep()

def MCalgo_Run_multiprocess_wrapper(MCcalc, nsteps):
    MCcalc.run(nsteps)
    return MCcalc

def MultiProcessReplicaRun(MCcalc_list, nsteps, pool):
    n_replicas = len(MCcalc_list)
    results = [
        pool.apply_async(
            MCalgo_Run_multiprocess_wrapper,(MCcalc_list[rep],
                                             nsteps)
        )
        for rep in range(n_replicas)
    ]
    return [res.get(timeout=1800) for res in results]

        
            
class TemperatureReplicaExchange:

    def __init__(self, model, kTs, configs, MCalgo):
        assert len(kTs) == len(configs)
        self.model = model
        self.kTs = kTs
        self.betas = 1.0/kTs
        self.n_replicas = len(kTs)
        self.MCreplicas = []
        self.accept_count = 0
        for i in range(self.n_replicas):
            self.MCreplicas.append(MCalgo(model, kTs[i], configs[i]))

    def Xtrial(self):
        # pick a replica
        rep = randrange(self.n_replicas - 1)
        delta = (self.betas[rep + 1] - self.betas[rep]) \
                *(self.MCreplicas[rep].energy -
                  self.MCreplicas[rep+1].energy)
        #print self.MCreplicas[rep].energy, self.model.energy(self.MCreplicas[rep].config)
                
        if delta < 0.0:
            # swap configs, energy
            tmp = self.MCreplicas[rep+1].config
            tmpe = self.MCreplicas[rep+1].energy
            self.MCreplicas[rep+1].config = self.MCreplicas[rep].config
            self.MCreplicas[rep+1].energy = self.MCreplicas[rep].energy
            self.MCreplicas[rep].config = tmp
            self.MCreplicas[rep].energy = tmpe
            self.accept_count += 1
        else:
            accept_probability = exp(-delta)
            #print accept_probability, "accept prob"
            if random() <= accept_probability:
                tmp = self.MCreplicas[rep+1].config
                tmpe = self.MCreplicas[rep+1].energy
                self.MCreplicas[rep+1].config = self.MCreplicas[rep].config
                self.MCreplicas[rep+1].energy = self.MCreplicas[rep].energy
                self.MCreplicas[rep].config = tmp
                self.MCreplicas[rep].energy = tmpe
                self.accept_count += 1
        
    def run(self, nsteps, attempt_frequency, pool):
        self.accept_count = 0
        outerloop = nsteps/attempt_frequency
        for i in range(outerloop):
            self.MCreplicas = MultiProcessReplicaRun(self.MCreplicas, attempt_frequency, pool)
            self.Xtrial()
            #self.configs = [MCreplica.config for MCreplica in self.MCreplicas]
        #print self.accept_count
        self.accept_count = 0
