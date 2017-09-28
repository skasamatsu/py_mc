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



class dft_zro2_spch(model):
    '''This class defines the DFT ZrO2 space charge model'''

    model_name = "dft_zro2_spch"
    
    def __init__(self, calcode, vasp_run, base_vaspinput, matcher, matcher_site,
                 queen):
        self.calcode = calcode
        self.matcher = matcher
        self.matcher_site = matcher_site
        self.drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
        self.queen = queen
        self.base_vaspinput = base_vaspinput
        self.vasp_run = vasp_run
        
    def energy(self, spch_config):
        ''' Calculate total energy of the space charge model'''
        
        structure = spch_config.structure
        
        #if len(spinel_config.calc_history) >= 20:
        #    print("truncate calc_history")
        #    del spinel_config.calc_history[0:10]
        #calc_history = spinel_config.calc_history
        #if calc_history:
        #    # Try to avoid doing dft calculation for the same structure.
        #    # Assuming that calc_history is a list of ComputedStructureEntries
        #    for i in range(len(calc_history)):
        #        if self.matcher.fit(structure, calc_history[i].structure):
        #            print("match found in history")
        #            return calc_history[i].energy
        #print("before poscar")
        poscar = Poscar(structure.get_sorted_structure())
        #print("before vaspinput")
        vaspinput = self.base_vaspinput
        vaspinput.update({'POSCAR':poscar})
        exitcode = self.vasp_run.submit(vaspinput, os.getcwd()+'/output')
        #print("vasp exited with exit code", exitcode)
        if exitcode !=0:
            print("something went wrong")
            sys.exit(1)
        #queen = BorgQueen(self.drone)
        self.queen.serial_assimilate('./output')
        results = self.queen.get_data()[-1]
        #calc_history.append(results)
        spch_config.structure = results.structure
        #print(results.energy)
        #sys.stdout.flush()
        
        return np.float64(results.energy)
        
    def xparam(self,spinel_config):
        '''Calculate number of B atoms in A sites'''
        
        asites = self.matcher_site.get_mapping(spinel_config.structure,
                                               spinel_config.Asite_struct)
        #print asites
        #print spinel_config.structure
        #print spinel_config.Asite_struct
        x = 0
        for i in asites:
            if spinel_config.structure.species[i] == Element(spinel_config.Bspecie):
                x += 1
        x /= float(len(asites))
        return x
        
            
    def trialstep(self, spch_config, energy_now):
        
        e0 = energy_now
        
        structure = spch_config.structure.copy()

        # Figure out where the vacancies are by comparing with base_structure
        filled_sites = self.matcher_site.get_mapping(spch_config.base_structure,
                                                     spch_config.structure)

        vac_struct = spch_config.base_structure.copy()
        vac_struct.remove_sites(filled_sites)
        assert len(vac_struct) == 10


        # choose one vacancy and one O atom randomly and flip
        vac_flip = rand.choice(len(vac_struct))
        Osites = structure.indices_from_symbol("O")
        O_flip = rand.choice(Osites)

        del structure[O_flip]
        structure.append(vac_struct[vac_flip])
        
        # Backup structure of previous step
        structure0 = spch_config.structure
        spch_config.structure = structure

        # do vasp calculation on structure
        e1 = self.energy(spch_config)

        # return old spch structure
        spch_config.structure = structure0
        
        # Simply pass new structure in dconfig to be used by newconfig():
        dconfig = structure

        dE = e1 - e0
        
        return dconfig, dE

    def newconfig(self, spch_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        spch_config.structure = dconfig
        return spch_config
        

        
class spch_config:
    '''This class defines the metal/ZrO2 space charge config'''

    def __init__(self, base_structure, cellsize, Vholder):
        self.base_structure = base_structure
        self.base_structure.make_supercell([cellsize, cellsize, 1])
        self.Osite_struct = self.base_structure.copy()
        self.Osite_struct.remove_species(["Pt", "Zr"])
        self.Zrsite_struct = self.base_structure.copy()
        self.Zrsite_struct.remove_species(["O", "Pt"])
        self.Vholder = Vholder
        self.structure = None 
        self.calc_history = []


    def prepare_Ovac(self, N_ovac):
        # Prepare a structure with N_ovac*N_Osites oxygen vacancies
        Osites = self.base_structure.indices_from_symbol("O")
        N_Vsites = int(N_ovac * len(Osites))
        Vacsites = rand.sample(Osites, N_Vsites)
        Vacsites.sort()
        Vacsites.reverse()
        self.structure = self.base_structure.copy()
        for site in Vacsites:
            self.structure.pop(site)

    def prepare_ordered(self):
        self.structure = self.base_structure.copy()
        
    #def __str__(self):
    #    s = ""
    #    for i in range(self.lenX):
    #        for j in range(self.lenY):
    #            if self.config[i,j] < 0:
    #                s += "-"
    #            else:
    #                s += "+"
    #        s += "\n"
    #    return s

def writeEandX(calc):
    with open("energy.out", "a") as f:
        f.write(str(calc.energy)+"\n")
        f.flush()
    with open("xparam.out", "a") as f:
        xparam = calc.model.xparam(calc.config)
        f.write(str(xparam)+"\n")
        f.flush()

def observables(MCcalc, outputfi):
    energy = MCcalc.energy
    energy2 = energy**2.0
    xparam = MCcalc.model.xparam(MCcalc.config)
    outputfi.write("\t".join([str(observable) for observable in [MCcalc.kT, energy, energy2, xparam]])+"\n")
    outputfi.flush()
    return np.array([energy, energy2, xparam])



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
    

    # prepare spinel_config
    cellsize = 2
    base_structure = Structure.from_file(os.path.join(os.path.dirname(__file__), "POSCAR")).get_primitive_structure(tolerance=0.001)
    config = spinel_config(base_structure, cellsize, "Mg", "Al")
    config.prepare_random()
    #config.prepare_ordered()
    
    configs = []
    for i in range(nreplicas):
         config.prepare_random()
         configs.append(copy.deepcopy(config)) 

    # prepare vasp spinel model
    vasprun = vasp_run_mpispawn("/home/i0009/i000900/src/vasp.5.3/vasp.spawnready.gamma", nprocs=nprocs_per_vasp, comm=comm)
    baseinput = VaspInput.from_directory("baseinput") #(os.path.join(os.path.dirname(__file__), "baseinput"))
    ltol=0.1
    matcher = StructureMatcher(ltol=ltol, primitive_cell=False, ignored_species=["Pt"])
    matcher_site = StructureMatcher(ltol=ltol, primitive_cell=False,
                                             allow_subset=True,
                                             comparator=FrameworkComparator(), ignored_species=["Pt"])
    drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
    queen = BorgQueen(drone)
    model = dft_spinel_mix(calcode="VASP", vasp_run=vasprun,  base_vaspinput=baseinput,
                           matcher=matcher, matcher_site=matcher_site, queen=queen)


    
    if worldrank == 0:
        print(config.structure)
        print(model.xparam(config))
    
    kTs = kB*np.array([500.0*1.1**i for i in range(nreplicas)])
    #configs = pickle.load(open("config.pickle","rb"))
    #configs = [copy.deepcopy(config) for i in range(nreplicas)]
    RXcalc = TemperatureRX_MPI(comm, CanonicalMonteCarlo, model, configs, kTs)
    RXcalc.reload()
    obs = RXcalc.run(nsteps=200, RXtrial_frequency=2, sample_frequency=1, observfunc=observables, subdirs=True)
    if worldrank == 0:
        for i in range(len(kTs)):
            print(kTs[i],"\t".join([str(obs[i,j]) for j in range(len(obs[i,:]))]))
        

