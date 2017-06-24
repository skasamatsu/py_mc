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
        # When done, run termination sequence if it exists
        # necessary when using multiprocess.queues to run
        # replicas of external programs
        try:
            self.model.terminate()
        except AttributeError:
            pass

def MCalgo_Run_multiprocess_wrapper(MCcalc, nsteps, outdir=None):
    if outdir:
        # create subdirectory and run there
        if not os.path.exists(outdir): os.mkdir(outdir)
        os.chdir(outdir)
        MCcalc.run(nsteps)
    else:
        MCcalc.run(nsteps)
    return MCcalc

def MultiProcessReplicaRun(MCcalc_list, nsteps, pool, subdirs=False):
    n_replicas = len(MCcalc_list)
    if subdirs:
        results = [
            pool.apply_async(
                MCalgo_Run_multiprocess_wrapper,(MCcalc_list[rep],
                                                 nsteps, str(rep))
            )
            for rep in range(n_replicas)
        ]
    else:
        results = [
            pool.apply_async(
                MCalgo_Run_multiprocess_wrapper,(MCcalc_list[rep],
                                                 nsteps)
            )
            for rep in range(n_replicas)
        ]
        results_list = [res.get(timeout=1800) for res in results]
        for result in results_list:
            if not result.sucessful():
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

    def __init__(self, model, kTs, configs, MCalgo, swap_algo=swap_configs):
        assert len(kTs) == len(configs)
        self.model = model
        self.kTs = kTs
        self.betas = 1.0/kTs
        self.n_replicas = len(kTs)
        self.MCreplicas = []
        self.accept_count = 0
        self.swap_algo = swap_algo
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
            self.MCreplicas, self.accept_count = self.swap_algo(self.MCreplicas,
                                                           rep, self.accept_count)
        else:
            accept_probability = exp(-delta)
            #print accept_probability, "accept prob"
            if random() <= accept_probability:
                self.MCreplicas, self.accept_count = self.swap_algo(self.MCreplicas,
                                                           rep, self.accept_count)
        
    def run(self, nsteps, attempt_frequency, pool, subdirs=False):
        self.accept_count = 0
        outerloop = nsteps/attempt_frequency
        for i in range(outerloop):
            self.MCreplicas = MultiProcessReplicaRun(self.MCreplicas, attempt_frequency, pool, subdirs)
            self.Xtrial()
            #self.configs = [MCreplica.config for MCreplica in self.MCreplicas]
        #print self.accept_count
        #self.accept_count = 0
