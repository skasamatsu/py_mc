# cython: profile=True

from math import exp
from random import random,randrange
import os, sys
import numpy as np
import sys


verylargeint = sys.maxsize
'''Defines base classes for Monte Carlo simulations'''


cdef class model:
    ''' This class defines a model whose energy equals 0 no matter the configuration, and the configuration
    never changes.
    This is a base template for building useful models.'''
    
    #model_name = None
    
    #def __init__(self):
        

    cpdef double energy(self, config):
        '''Calculate energy of configuration: input: config'''
        return 0.0

    def trialstep(self, config, energy):
        '''Define a trial step on config. Returns dconfig, which can contain the minimal information for
        constructing the trial configuration from config to be used in newconfig(). Make sure that
        config is the same upon entry and exit'''
        dE = 0.0
        dconfig = None
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

def binning(x, nlevels):
    error_estimate = []
    x = np.array(x)
    assert 2**nlevels*30 < len(x)
    throwout = len(x)%(2**nlevels)
    if throwout != 0:
        # The number of measurements must be divisible by 2**nlevels
        # If not, throw out initial measurements
        x = x[throwout:]
    error_estimate.append(np.sqrt(np.var(x, ddof=1)/len(x)))
    for lvl in range(1,nlevels):     
        x_tmp = x
        x = (x_tmp[0::2] + x_tmp[1::2])/2.0
        error_estimate.append(np.sqrt(np.var(x, ddof=1)/len(x)))
    return error_estimate

empty_array = np.array([])
#@profile
def obs_encode(*args):
    #nargs = np.array([len(args)])
    #args_length_list = []
    cdef double [:] obs_array = empty_array
    for arg in args:
        # Inelegant way to make everything a 1D array
        arg = np.array([arg])
        arg = arg.ravel()
        obs_array = np.concatenate((obs_array, arg))
        #args_length_list.append(len(arg))
    #args_length_array = np.array(args_length_list)
    #args_info = np.concatenate((nargs, args_length_array))
    return obs_array

def args_info(*args):
    nargs = np.array([len(args)])
    args_length_list = []
    #obs_array = np.array([])
    for arg in args:
        # Inelegant way to make everything a 1D array
        arg = np.array([arg])
        arg = arg.ravel()
        #obs_array = np.concatenate((obs_array, arg))
        args_length_list.append(len(arg))
    args_length_array = np.array(args_length_list)
    args_info = np.concatenate((nargs, args_length_array))
    return args_info


def obs_decode(args_info, obs_array):
    nargs = args_info[0]
    args_length_array = args_info[1:nargs+1]
    args = []
    idx = 0
    for i in range(nargs):
        length = args_length_array[i]
        if length == 1:
            args.append(obs_array[idx])
        else:
            args.append(obs_array[idx:idx+length])
        idx += length
    return args

def make_observefunc(logfunc,*multiDfuncs):
    def observefunc(calc_state, outputfi):
        obs_log = logfunc(calc_state)
        outputfi.write(str(calc_state.kT)+"\t")
        if hasattr(obs_log, '__getitem__'):
            outputfi.write("\t".join([str(observable) for observable in obs_log])+"\n")
        else:
            outputfi.write(str(obs_log)+"\n")
        obs_ND = []
        for func in multiDfuncs:
            obs_ND.append(func(calc_state))
        return obs_encode(*obs_log, *obs_ND)
    return observefunc

        
cdef class observer_base:
    def obs_info(self, calc_state):
        obs_log = self.logfunc(calc_state)
        obs_ND = []
        obs_save = self.savefuncs(calc_state)
        if obs_save != None:
            if isinstance(obs_save, tuple):
                for obs in obs_save:
                    obs_ND.append(obs)

            else:
                obs_ND.append(obs_save)
        return args_info(*obs_log, *obs_ND)

    cpdef logfunc(self, calc_state):
        return calc_state.energy,
    cpdef savefuncs(self, calc_state):
        return None
    cpdef observe(self, calc_state, outputfi,  lprint=True):
        obs_log = np.atleast_1d(self.logfunc(calc_state))
        if lprint:
            outputfi.write(str(calc_state.kT)+"\t"+
                           "\t".join([str(x) for x in obs_log])+"\n")

        obs_save = self.savefuncs(calc_state)
        if obs_save != None:
            obs_save = np.atleast_1d(obs_save)
            obs_save.ravel()
            return np.concatenate((obs_log, obs_save))
        else:
            return obs_log
        
class CanonicalMonteCarlo:
    def __init__(self, model, kT, config, grid=0):
        self.model = model
        self.config = config
        self.kT = kT
        self.grid = grid
    #@profile
    def MCstep(self):
        dconfig, dE  = self.model.trialstep(self.config, self.energy)
        if self.energy == float("inf"):
            self.config = self.model.newconfig(self.config, dconfig)
            self.energy = dE
        elif dE < 0.0:
            self.config = self.model.newconfig(self.config, dconfig)
            self.energy += dE
            #print "trial accepted"
        else:
            accept_probability = exp(-dE/self.kT)
            if random() <= accept_probability:
                self.config = self.model.newconfig(self.config, dconfig)
                self.energy += dE
                #print "trial accepted"

    def run(self, int nsteps, long sample_frequency=verylargeint, long print_frequency=verylargeint, observer=observer_base()):
        cdef int i
            
        observables = 0.0
        nsample = 0
        self.energy = self.model.energy(self.config)
        output = open("obs.dat", "a")
        if hasattr(observer.observe(self,open(os.devnull,"w")),'__add__'):
            observe = True
        else:
            observe = False

        for i in range(1,nsteps+1):
            self.MCstep()
            #sys.stdout.flush()
            if  observe and i%sample_frequency == 0:
                obs_step = observer.observe(self, output, i%print_frequency==0)
                observables += obs_step
                nsample += 1
        if nsample > 0:
            observables /= nsample
            args_info = observer.obs_info(self)
            return obs_decode(args_info, observables)
        else:
            return None
        



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
    
