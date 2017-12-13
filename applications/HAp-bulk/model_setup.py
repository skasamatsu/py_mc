import numpy as np
import random as rand
import sys, os
import copy
import pickle
#from mpi4py import MPI

from pymatgen import Lattice, Structure, Element, PeriodicSite
from pymatgen.io.vasp import Poscar, VaspInput
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from mc import model

def gauss(x, x0, sigma):
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x-x0)/sigma, 2.)/2.)


class dft_HAp(model):
    '''This class defines the DFT HAp space charge model'''

    model_name = "dft_HAp"
    
    def __init__(self, calcode, vasp_run, base_vaspinput, # matcher, matcher_site,
                 queen, selective_dynamics=None):
        self.calcode = calcode
        #self.matcher = matcher
        #self.matcher_site = matcher_site
        self.drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
        self.queen = queen
        self.base_vaspinput = base_vaspinput
        self.vasp_run = vasp_run
        self.selective_dynamics = selective_dynamics
        
    def energy(self, HAp_config):
        ''' Calculate total energy of the space charge model'''
        
        structure = HAp_config.structure.get_sorted_structure()
        
        if self.selective_dynamics:
            seldyn_arr = [[True,True,True] for i in range(len(structure))]
            for specie in self.selective_dynamics:
                indices = structure.indices_from_symbol(specie)
                for i in indices:
                    seldyn_arr[i] = [False, False, False]
        else:
            seldyn_arr = None        
            
        poscar = Poscar(structure=structure,selective_dynamics=seldyn_arr)
        vaspinput = self.base_vaspinput
        vaspinput.update({'POSCAR':poscar})
        exitcode = self.vasp_run.submit(vaspinput, os.getcwd()+'/output')
        if exitcode !=0:
            print("something went wrong")
            sys.exit(1)
        queen = BorgQueen(self.drone)
        queen.serial_assimilate('./output')
        results = queen.get_data()[-1]
        HAp_config.structure = results.structure
        
        return np.float64(results.energy)
        
    # def xparam(self,spinel_config):
    #     '''Calculate number of B atoms in A sites'''
        
    #     asites = self.matcher_site.get_mapping(spinel_config.structure,
    #                                            spinel_config.Asite_struct)
    #     #print asites
    #     #print spinel_config.structure
    #     #print spinel_config.Asite_struct
    #     x = 0
    #     for i in asites:
    #         if spinel_config.structure.species[i] == Element(spinel_config.Bspecie):
    #             x += 1
    #     x /= float(len(asites))
    #     return x
        
            
    def trialstep(self, HAp_config, energy_now):
        
        e0 = energy_now

        # Back up structure and latgas_rep
        structure0 = HAp_config.structure
        latgas_rep0 = HAp_config.latgas_rep.copy()

        if np.count_nonzero(latgas_rep0==0) == 0 or rand.random() < 0.5:
            # Flip an OH
            OH_ids = np.where(abs(latgas_rep0)==1)[0]
            flip_id = rand.choice(OH_ids)
            HAp_config.latgas_rep[flip_id] *= -1
        else:
            # Exchange V with OH or O
            V_ids = np.where(latgas_rep0 == 0)[0]
            ex_id = rand.choice(V_ids)
            other_ids = np.where(latgas_rep0 != 0)[0]
            ex_other_id = rand.choice(other_ids)
            HAp_config.latgas_rep[ex_id], HAp_config.latgas_rep[ex_other_id] \
                = HAp_config.latgas_rep[ex_other_id], HAp_config.latgas_rep[ex_id]
            
        HAp_config.set_latgas()     

        # do vasp calculation on structure
        e1 = self.energy(HAp_config)

        # return old structure
        structure = HAp_config.structure
        HAp_config.structure = structure0
        latgas_rep = HAp_config.latgas_rep
        HAp_config.latgas_rep = latgas_rep0
        
        # Simply pass new structure and latgas_rep  in dconfig to be used by newconfig():
        dconfig = structure, latgas_rep

        dE = e1 - e0
        
        return dconfig, dE

    def newconfig(self, HAp_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        HAp_config.structure, HAp_config.latgas_rep = dconfig
        return HAp_config
        

        
class HAp_config:
    '''This class defines the HAp config with lattice gas mapping'''

    def __init__(self, base_structure, cellsize=[1,1,1]):
        self.cellsize = cellsize
        self.base_structure = base_structure
        site_centers = [[0.,0.,0.25],[0.,0.,0.75]]
        self.latgas_rep = np.ones(cellsize[0]*cellsize[1]*cellsize[2]*len(site_centers),dtype=int)
        site_centers = np.array(site_centers)
        site_centers_sc = np.zeros(np.prod(cellsize)*site_centers.shape[0],3, dtype=float)
        idx = 0
        for i in range(cellsize[0]):
            for j in range(cellsize[1]):
                for k in range(cellsize[2]):     
                    site_centers_sc[idx] = site_centers + np.array([i,j,k])
                    idx += 1

        self.site_centers = site_centers_sc
        self.base_structure.make_supercell([cellsize[0], cellsize[1], cellsize[2]])
        self.supercell = self.base_structure.lattice.matrix
        group_dict = {
            0:[],
            (1,0):[["O",np.array([0.0,0.0,-0.915/2])],["H",np.array([0.0,0.0,0.915/2])]],
            (1,1):[["O",np.array([0.0,0.0,0.915/2])],["H",np.array([0.0,0.0,-0.915/2])]],
            2: ["O",np.array([0.0,0.0,0.0])]
        }
        invSuper = np.linalg.inv(self.supercell)
        for key in group_dict.keys():
            for atom in group_dict[key]:
                atom[1] = np.dot(atom[1], invSuper)
        self.group_dict = group_dict

    def set_latgas(self,latgas_rep=self.latgas_rep):
        self.latgas_rep = latgas_rep, dtype=int
        numsites = self.latgas_rep.shape[0]
        assert numsites == self.site_centers.shape[0]
        self.structure = copy.deepcopy(self.base_structure)
        for isite in range(numsites):
            gid = latgas_rep[isite]
            for atom in range(len(self.group_dict[gid])):
                self.structure.append(
                    atom[0],
                    atom[1] + self.site_center[isite],
                    properties={"velocities":[0,0,0]}
                    )
                    
        
    def prepare_Ovac(self):
        # Prepare a structure with N_ovac*N_Osites oxygen vacancies
        Osites = self.base_structure.indices_from_symbol("O")
        N_Vsites = int(self.N_Ovac * len(Osites))
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
    
# def writeEandX(calc):
#     with open("energy.out", "a") as f:
#         f.write(str(calc.energy)+"\n")
#         f.flush()
#     with open("xparam.out", "a") as f:
#         xparam = calc.model.xparam(calc.config)
#         f.write(str(xparam)+"\n")
#         f.flush()



def observables(MCcalc, outputfi):
    energy = MCcalc.energy
    nup = np.count_nonzero(MCcalc.config.latgas_rep==1)
    ndown = np.count_nonzero(MCcalc.config.latgas_rep==-1)
    tot_pol = nup - ndown
    #energy2 = energy**2.0
    #xparam = MCcalc.model.xparam(MCcalc.config)
    outputfi.write("\t".join([str(observable) for observable in [MCcalc.kT, energy, nup, ndown, tot_pol, tot_pol**2]])+"\n")
    outputfi.flush()
    return [MCcalc.kT, energy, nup, ndown, tot_pol, tot_pol**2]




