from math import exp
from random import random

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
        constructing the trial configuration from config to be used in newconfig().'''
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
    

        
        