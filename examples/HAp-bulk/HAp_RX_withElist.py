import numpy as np
import random as rand
import sys, os
import copy
import pickle
from mpi4py import MPI

from pymatgen import Lattice, Structure, Element
from pymatgen.io.vasp import Poscar, VaspInput
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
#from mc.applications.dft_spinel_mix.dft_spinel_mix import dft_spinel_mix, spinel_config
from applications.dft_spinel_mix.run_vasp_mpi import vasp_run_mpispawn
from mc import model, CanonicalMonteCarlo, MultiProcessReplicaRun, TemperatureReplicaExchange
from mc_mpi import TemperatureRX_MPI

from model_setup import *


if __name__ == "__main__":
    args = sys.argv
    kB = 8.6173e-5
    nreplicas = int(args[1] )
    nprocs_per_vasp = int(args[2])
    commworld = MPI.COMM_WORLD
    worldrank = commworld.Get_rank()
    worldprocs = commworld.Get_size()
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
    

    # prepare config
    cellsize = [1,1,3]
    base_structure = Structure.from_file(os.path.join(os.path.dirname(__file__), "POSCAR"))#.get_primitive_structure(tolerance=0.001)
    config = HAp_config(base_structure=base_structure, cellsize=cellsize)
    config.set_latgas()
    
    configs = []
    for i in range(nreplicas):
         configs.append(copy.deepcopy(config)) 

    # prepare vasp spinel model
    vasprun = vasp_run_mpispawn("/home/i0009/i000900/src/vasp.5.3/vasp.spawnready.gamma", nprocs=nprocs_per_vasp, comm=comm)
    baseinput = VaspInput.from_directory("baseinput") #(os.path.join(os.path.dirname(__file__), "baseinput"))
    ltol=0.1
#    matcher = StructureMatcher(ltol=ltol, primitive_cell=False, ignored_species=["Pt"])
    matcher_base = StructureMatcher(ltol=ltol, primitive_cell=False,stol=0.5,
                                             allow_subset=True)#,
                                             #comparator=FrameworkComparator(), ignored_species=["Pt","Zr"])
    drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
    queen = BorgQueen(drone)

    energy_lst = pickle.load(open("energy_reps.pickle", "rb"))
    reps = pickle.load(open("latgas_reps.pickle", "rb"))
    
    model = energy_lst_HAp(calcode="VASP", vasp_run=vasprun,  base_vaspinput=baseinput, matcher_base=matcher_base,
                            queen=queen, reps=reps, energy_lst=energy_lst)
    
    if worldrank == 0:
        print(config.structure)
        #print(model.xparam(config))
    
    kTs = kB*np.array([500.0*1.2**i for i in range(nreplicas)])
    #configs = pickle.load(open("config.pickle","rb"))
    #configs = [copy.deepcopy(config) for i in range(nreplicas)]
    RXcalc = TemperatureRX_MPI(comm, CanonicalMonteCarlo, model, configs, kTs)
    #RXcalc.reload()
    for nstep in range(10):
        obs = RXcalc.run(nsteps=1000, RXtrial_frequency=2, sample_frequency=1, observfunc=observables, subdirs=True)
        if worldrank == 0:
            with open("step"+str(nstep)+".dat", "w") as f:
                for i in range(len(kTs)):
                    f.write("\t".join([str(obs[i,j]) for j in range(len(obs[i,:]))])+"\n")
        

