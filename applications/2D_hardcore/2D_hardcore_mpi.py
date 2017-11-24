import numpy as np
import random as rand
import sys
import copy
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpi4py import MPI

from mc import CanonicalMonteCarlo, grid_1D
from mc_mpi import ParallelMC
from model_setup import *

def parabola(x,a,b,c):
    return a*x**2.0 + b*x + c

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    procs = comm.Get_size()
    myrank = comm.Get_rank()
    L = 1.0
    nu = 5
    eos = open('eos.dat','a')
    for nu in np.arange(5.4,6.4,0.02):
        d = 1.0/14.0
        rdisc = d*(1.0-2.0**(nu-8))*0.5
        Ndisc = 224
        dr = 0.0005
        maxstep = d - rdisc*2.0
        kT = 1
        A0 = 3.0**0.5*0.5*(rdisc*2.0)**2.0*Ndisc

        grid = grid_1D(dr, dr, L/4.0)
        config = HC_2D_config(L, Ndisc, rdisc)
        model = HC_2D(maxstep)
        config.prepare_metropolis()
        #print(config.coors)
        #print(g_r(config, grid))
        calc = CanonicalMonteCarlo(model, kT, config, grid=grid)
        #plot_fig(calc, 100000)
        colors = ["blue", "red", "green", "magenta"]
        config.prepare_metropolis()
        configs = [copy.deepcopy(config) for i in range(procs)]
        kTs = [kT for i in range(procs)]
        parallel_calc = ParallelMC(comm, CanonicalMonteCarlo, model, configs, kTs, grid)
        obs_par = parallel_calc.run(10000,
                                    sample_frequency=0)
        obs_par = parallel_calc.run(200000,sample_frequency=100,observefunc=observables)
        obs = np.sum(obs_par, axis=0)/procs
        rdisc2_id = int(2*rdisc//dr)
        popt,pcov = curve_fit(parabola,grid.x[rdisc2_id+1:rdisc2_id+8],obs[rdisc2_id+1:rdisc2_id+8])
        #print(popt)
        if myrank == 0:
            g_sigma = parabola(2*rdisc,*popt)
            eos_rhs = 1.0 + Ndisc*2*rdisc*g_sigma
            print("g(sigma) = " + str(g_sigma))
            eos.write(str(nu)+'\t'+str(A0)+'\t'+str(eos_rhs)+'\n')
            eos.flush()
            plt.cla()
            plt.plot(grid.x, obs)
            plt.plot(grid.x[rdisc2_id:rdisc2_id+8], parabola(grid.x[rdisc2_id:rdisc2_id+8],*popt))
            #plt.plot(x_cs, cs(x_cs), label='average')
            #plt.plot(grid.x, obs/4)
            #plt.show()
            plt.savefig('nu'+str(nu)+'.png')
            #print(config.coors)
    
    
    '''calc = Canonical
    
    J = -1.0
    kT = abs(J) * 1.0
    size = 10
    eqsteps = 100000
    mcsteps = 1000000
    sample_frequency = size*size
    config = ising2D_config(size,size)
    config.prepare_random()
    model = ising2D(J)

    for kT in [5]: #np.arange(5, 0.5, -0.05):
        energy_expect = 0
        magnet_expect = 0
        
        kT = abs(J)*kT        

        #print config        
        calc = CanonicalMonteCarlo(model, kT, config)
        calc.run(eqsteps)

        mcloop = mcsteps//sample_frequency
        for i in range(mcloop):
            calc.run(sample_frequency)
            #print model.energy(config), model.magnetization(config)
            current_config = calc.config
            energy_expect += model.energy(current_config)
            magnet_expect += abs(model.magnetization(current_config))
        print(kT, energy_expect/mcloop, magnet_expect/mcloop)
        sys.stdout.flush()
    #calc.run(100000)
    #print config
    
        
'''
