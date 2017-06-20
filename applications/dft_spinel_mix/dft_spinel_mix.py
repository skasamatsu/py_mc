import numpy as np
import random as rand
import sys, os
import copy
import cPickle

from mc import model, CanonicalMonteCarlo
from pymatgen import Lattice, Structure, Element
from pymatgen.io.vasp import Poscar, VaspInput
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from run_vasp import vasp_run

class dft_spinel_mix(model):
    '''This class defines the DFT spinel model'''

    model_name = "dft_spinel"
    
    def __init__(self, calcode, ltol, vasp_run_cmd):
        self.calcode = calcode
        self.matcher = StructureMatcher(ltol=ltol, primitive_cell=False)
        self.drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
        self.queen = BorgQueen(self.drone)
        self.vasp_run_cmd = vasp_run_cmd
        #self.base_structure = Poscar.from_file("MgAl2O4.vasp")
        #self.base_structure.make_supercell(
        #    [supercellsize,supercellsize,supercellsize])

    def energy(self, spinel_config):
        ''' Calculate total energy of the spinel model'''
        structure = spinel_config.structure
        #inputdir = spinel_config.inputdir
        calc_history = spinel_config.calc_history
        if calc_history:
            # Try to avoid doing dft calculation for the same structure.
            # Assuming that calc_history is a list of ComputedStructureEntries
            for i in range(len(calc_history)):
                if self.matcher.fit(structure, calc_history[i].structure):
                    print "match found in history"
                    return calc_history[i].energy
        poscar = Poscar(structure)
        vaspinput = VaspInput.from_directory('baseinput')
        vaspinput.update({'POSCAR':poscar})
        p = vasp_run(vaspinput, 'output', self.vasp_run_cmd)
        exitcode = p.wait()
        print "vasp exited with exit code", exitcode
        if exitcode !=0:
            print "something went wrong"
            sys.exit(1)
        queen.serial_assimilate('./output')
        results = queen.get_data()
        calc_history.append(results)
        return results.energy
        

    def xparam(self,spinel_config):
        '''Calculate number of B atoms in A sites'''
        asites = spinel_config.Asites
        x = 0
        for i in asites:
            if spinel_config.structure.species[i] == Element(spinel_config.Bspecie):
                x += 1
        x /= float(len(asites))
        return x

    def trialstep(self, spinel_config):
        # Get energy for the current step (vasp shouldn't run except for first step
        # because result should be in the calc_history
        e0 = self.energy(spinel_config)
        
        # choose one A atom and one B atom randomly and flip
        structure = copy.deepcopy(spinel_config.structure)
        Aspecie = spinel_config.Aspecie
        Bspecie = spinel_config.Bspecie
        Apos = structure.indices_from_symbol(Aspecie)
        Bpos = structure.indices_from_symbol(Bspecie)
        A_flip = rand.choice(Apos)
        B_flip = rand.choice(Bpos)

        structure[A_flip] = Bspecie
        structure[B_flip] = Aspecie

        # Backup structure of previous step
        structure0 = spinel_config.structure
        spinel_config.structure = structure

        # do vasp calculation on structure
        e1 = self.energy(spinel_config)

        # return old spinel structure
        spinel_config.structure = structure0
        
        # Simply pass new structure in dconfig to be used by newconfig():
        dconfig = structure

        dE = e1 - e0
        
        return dconfig, dE

    def newconfig(self, spinel_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        spinel_config.structure = dconfig
        return spinel_config
        

        
class spinel_config:
    '''This class defines the disordered spinel model configuration'''

    def __init__(self, cellsize, Aspecie, Bspecie):
        self.base_structure = Structure.from_file(os.path.join(os.path.dirname(__file__), "POSCAR"))
        self.base_structure.make_supercell([cellsize, cellsize, cellsize])
        self.Asites = self.base_structure.indices_from_symbol("Mg")
        self.Bsites = self.base_structure.indices_from_symbol("Al")
        self.Aspecie = Aspecie
        self.Bspecie = Bspecie
        self.base_structure["Mg"] = Aspecie
        self.base_structure["Al"] = Bspecie
        self.structure = None 
        self.calc_history = []


    def prepare_random(self):
        # Prepare a structure where 1/2 of A sites are exchanged with B sites randomly
        Asites = self.Asites
        Bsites = self.Bsites
        flipsites = len(Asites)/2
        Aflip = rand.sample(Asites, flipsites)
        Bflip = rand.sample(Bsites, flipsites)
        self.structure = copy.deepcopy(self.base_structure)
        for i in range(flipsites):
            self.structure[Aflip[i]] = self.Bspecie
            self.structure[Bflip[i]] = self.Aspecie
        
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


if __name__ == "__main__":
    kB = 8.6173e-5
    cellsize = 1
    eqsteps = 100
    mcsteps = 10000
    sample_frequency = 100
    config = spinel_config(cellsize, "Mg", "Al")
    config.prepare_random()
    print config.structure
    vasp_run_cmd = "mpijob /home/issp/vasp/vasp.5.3.5/bin/vasp.gamma"
    model = dft_spinel_mix(calcode="VASP", ltol=0.2, vasp_run_cmd=vasp_run_cmd)

    for T in [1500]:
        energy_expect = 0
        xparam_expect = 0
        
        kT = kB*T

        #print config        
        #calc = CanonicalMonteCarlo(model, kT, config)
#        calc.run(eqsteps)
#        cPickle.dump(config, open("spinel_config.pickle", "wb"))

        mcloop = mcsteps/sample_frequency
        #for i in range(mcloop):
            #calc.run(sample_frequency)
            #print model.energy(config), model.magnetization(config)
            #energy_expect += model.energy(config)
            #magnet_expect += abs(model.magnetization(config))
        #print kT, energy_expect/mcloop, xparam_expect/mcloop
        print model.xparam(config)
        sys.stdout.flush()
    #calc.run(100000)
    #print config
    
        
