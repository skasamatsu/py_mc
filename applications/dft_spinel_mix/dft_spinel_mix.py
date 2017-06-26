import numpy as np
import random as rand
import sys, os
import copy
import cPickle
import multiprocessing as mp

from pymatgen import Lattice, Structure, Element
from pymatgen.io.vasp import Poscar, VaspInput
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
#from mc.applications.dft_spinel_mix.dft_spinel_mix import dft_spinel_mix, spinel_config
import applications.dft_spinel_mix.run_vasp as rvasp
from mc import model, CanonicalMonteCarlo, MultiProcessReplicaRun

mp.allow_connection_pickling()

class dft_spinel_mix(model):
    '''This class defines the DFT spinel model'''

    model_name = "dft_spinel"
    
    def __init__(self, calcode, vasp_run, base_vaspinput, matcher, matcher_site,
                 queen):
        self.calcode = calcode
        self.matcher = matcher
        self.matcher_site = matcher_site
        self.drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
        self.queen = queen
        self.base_vaspinput = base_vaspinput
        self.vasp_run = vasp_run
        
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
        print "before poscar"
        poscar = Poscar(structure.get_sorted_structure())
        print "before vaspinput"
        vaspinput = self.base_vaspinput
        vaspinput.update({'POSCAR':poscar})
        exitcode = self.vasp_run.submit(vaspinput, os.getcwd()+'/output')
        print "vasp exited with exit code", exitcode
        if exitcode !=0:
            print "something went wrong"
            sys.exit(1)
        #queen = BorgQueen(self.drone)
        self.queen.serial_assimilate('./output')
        results = self.queen.get_data()[-1]
        calc_history.append(results)
        spinel_config.structure = results.structure
        print results.energy
        sys.stdout.flush()
        
        return 0#results.energy
        
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
        
            
    def trialstep(self, spinel_config):
        # Get energy for the current step (vasp shouldn't run except for first step
        # because result should be in the calc_history
        
        e0 = self.energy(spinel_config)
        
        # choose one A atom and one B atom randomly and flip
        structure = spinel_config.structure.copy()
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

    def __init__(self, base_structure, cellsize, Aspecie, Bspecie):
        self.base_structure = base_structure
        self.base_structure.make_supercell([cellsize, cellsize, cellsize])
        self.Asite_struct = self.base_structure.copy()
        self.Asite_struct.remove_species(["O", "Al"])
        self.Bsite_struct = self.base_structure.copy()
        self.Bsite_struct.remove_species(["O", "Mg"])
        self.Aspecie = Aspecie
        self.Bspecie = Bspecie
        self.base_structure["Mg"] = Aspecie
        self.base_structure["Al"] = Bspecie
        self.structure = None 
        self.calc_history = []


    def prepare_random(self):
        # Prepare a structure where 1/2 of A sites are exchanged with B sites randomly
        Asites = self.base_structure.indices_from_symbol("Mg")
        Bsites = self.base_structure.indices_from_symbol("Al")
        flipsites = len(Asites)/2
        Aflip = rand.sample(Asites, flipsites)
        Bflip = rand.sample(Bsites, flipsites)
        self.structure = self.base_structure.copy()
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
    eqsteps = 300
    mcsteps = 10000
    sample_frequency = 100

    # prepare spinel_config
    cellsize = 1
    base_structure = Structure.from_file(os.path.join(os.path.dirname(__file__), "POSCAR"))#.get_primitive_structure(tolerance=0.001)
    config = spinel_config(base_structure, cellsize, "Mg", "Al")
    config.prepare_random()
    print config.structure

    # prepare queue and qwatcher for submitting vasp jobs
    queue = mp.Manager().Queue()
    nvaspruns = 3
    n_mpiprocs = 72
    n_ompthreads = 1
    synctime = 10
    qwatcher = mp.Process(target=rvasp.vasp_bulkjob_qwatcher,
                          args=(queue, "/home/issp/vasp/vasp.5.3.5/bin/vasp.gamma",
                                nvaspruns, n_mpiprocs, n_ompthreads, synctime)
                          )

    # prepare dft_spinel_mix model
    vasprun = rvasp.vasp_run_use_queue(queue)
    #rvasp.vasp_run("mpijob /home/issp/vasp/vasp.5.3.5/bin/vasp.gamma")# rvasp.vasp_run_use_queue(queue)
    baseinput = VaspInput.from_directory("baseinput") #(os.path.join(os.path.dirname(__file__), "baseinput"))
    ltol=0.2
    matcher = StructureMatcher(ltol=ltol, primitive_cell=False)
    matcher_site = StructureMatcher(ltol=ltol, primitive_cell=False,
                                             allow_subset=True,
                                             comparator=FrameworkComparator())
    drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
    queen = BorgQueen(drone)
    model = dft_spinel_mix(calcode="VASP", vasp_run=vasprun,  base_vaspinput=baseinput,
                           matcher=matcher, matcher_site=matcher_site, queen=queen)

    print model.xparam(config)

    # Prepare pool of workers for Monte Carlo replicas
    nreplicas = 3  
    pool = mp.Pool(processes=nreplicas)
    #sys.exit()

    # Start qwatcher
    qwatcher.start()
    
    for T in [1500]:
        energy_expect = 0
        xparam_expect = 0
        
        kT = kB*T
        calc_list = []
        for i in range(nreplicas):
            calc_list.append(CanonicalMonteCarlo(model, kT, copy.deepcopy(config)))
        #calc_list[0].run(2)
        #print config
        xparam_out = open("xparam.out", "w")
        for i in range(10):
            print "before MPRR"
            calc_list = MultiProcessReplicaRun(calc_list, 1, pool, True)
            xparam_out.write("\t".join([str(model.xparam(calc.config)) for calc in calc_list]))
            xparam_out.flush()
        #calc.run(eqsteps)
        #cPickle.dump(config, open("spinel_config.pickle", "wb"))

        #mcloop = mcsteps/sample_frequency
        #for i in range(mcloop):
            #calc.run(sample_frequency)
            #print model.energy(config), model.magnetization(config)
            #energy_expect += model.energy(config)
            #magnet_expect += abs(model.magnetization(config))
        #print kT, energy_expect/mcloop, xparam_expect/mcloop
        #print model.xparam(config)
        #sys.stdout.flush()
    #calc.run(100000)
    #print config
    
        
