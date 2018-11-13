import os, sys, shutil
from pymatgen.io.vasp import Poscar, Kpoints, Potcar, VaspInput
from pymatgen import Structure
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from ctypes import *
from mpi4py import MPI
from timeit import default_timer as timer
import subprocess
import time
import numpy as np

class test_runner(object):
    def submit(self, structure, output_dir, seldyn_arr=None):
        poscar = Poscar(structure=structure,selective_dynamics=seldyn_arr)
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
                                    
        poscar.write_file(output_dir+"/POSCAR.vasp")
        return 0, structure
    

class vasp_runner(object):
    def __init__(self, base_input_dir, path_to_vasp, nprocs_per_vasp, comm, perturb=0):
        self.base_vasp_input = VaspInput.from_directory(base_input_dir)
        self.path_to_vasp = path_to_vasp
        self.nprocs_per_vasp = nprocs_per_vasp
        self.comm = comm
        self.vasp_run = vasp_run_mpispawn(path_to_vasp, nprocs_per_vasp, comm)
        self.drone = SimpleVaspToComputedEntryDrone(inc_structure=True)
        self.queen = BorgQueen(self.drone)
        self.perturb = perturb
        
    def submit(self, structure, output_dir, seldyn_arr=None):
        if self.perturb:
            structure.perturb(self.perturb)
        poscar = Poscar(structure=structure,selective_dynamics=seldyn_arr)
        vaspinput = self.base_vasp_input
        vaspinput.update({'POSCAR':poscar})
        self.vasp_run.submit(vaspinput, output_dir)
        #queen = BorgQueen(self.drone)
        self.queen.serial_assimilate(output_dir)
        results = self.queen.get_data()[-1]
        return np.float64(results.energy), results.structure

class vasp_runner_multistep(object):
    def __init__(self, base_input_dirs, path_to_vasp, nprocs_per_vasp, comm, perturb=0):
        self.vasp_runners = []
        assert len(base_input_dirs) > 1
        self.vasp_runners.append(vasp_runner(base_input_dirs[0], path_to_vasp, nprocs_per_vasp, comm, perturb))
        for i in range(1,len(base_input_dirs)):
            self.vasp_runners.append(vasp_runner(base_input_dirs[i], path_to_vasp, nprocs_per_vasp, comm, perturb=0))

    def submit(self, structure, output_dir, seldyn_arr=None):
        energy, newstructure = self.vasp_runners[0].submit(structure, output_dir, seldyn_arr)
        for i in range(1, len(self.vasp_runners)):
            energy, newstructure = self.vasp_runners[i].submit(newstructure, output_dir, seldyn_arr)
        return energy, newstructure
            

def submit_bulkjob(vasprundirs, path_to_vasp, n_mpiprocs, n_ompthreads):
    joblist = open("joblist.txt", "w")
    if n_ompthreads != 1:
        progtype = "H"+str(n_ompthreads)
    else:
        progtype = "M"
    for vasprundir in vasprundirs:
        joblist.write(
            ";".join([path_to_vasp, str(n_mpiprocs), progtype, vasprundir]) + "\n")
    stdout = open("stdout.log", "w")
    stderr = open("stderr.log", "w")
    stdin = open(os.devnull, "r")
    joblist.flush()
    start = timer()
    p = subprocess.Popen("bulkjob ./joblist.txt", stdout=stdout, stderr=stderr, stdin=stdin,
                         shell=True)
    exitcode = p.wait()
    end = timer()
    print("it took ",end-start," secs. to start vasp and finish")
    sys.stdout.flush()
    return exitcode
    

class vasp_run_mpibulkjob:
    def __init__(self, path_to_spawn_ready_vasp, nprocs, comm):
        self.path_to_vasp = path_to_spawn_ready_vasp
        self.nprocs = nprocs
        self.comm = comm
        self.commsize = comm.Get_size()
        self.commrank = comm.Get_rank()
    def submit(self, VaspInput, output_dir):
        VaspInput.write_input(output_dir=output_dir)
        vasprundirs = self.comm.gather(output_dir, root=0)
        exitcode = 1
        if self.commrank == 0:
            exitcode = np.array([submit_bulkjob(
                vasprundirs, self.path_to_vasp, self.nprocs, 1)])
            for i in range(1, self.commsize):
                self.comm.Isend([exitcode, MPI.INT], dest=i, tag=i)
            
        else:
            exitcode = np.array([0])
            while not self.comm.Iprobe(source=0, tag=self.commrank):
                time.sleep(0.2)      
            self.comm.Recv([exitcode, MPI.INT], source=0, tag=self.commrank)
        return exitcode[0]




class vasp_run_mpispawn:
    def __init__(self, path_to_spawn_ready_vasp, nprocs, comm):
        self.path_to_vasp = path_to_spawn_ready_vasp
        self.nprocs = nprocs
        self.comm = comm
        self.commsize = comm.Get_size()
        self.commrank = comm.Get_rank()
        commworld = MPI.COMM_WORLD
        self.worldrank = commworld.Get_rank()

    def submit(self, VaspInput, output_dir, rerun=2):
        VaspInput.write_input(output_dir=output_dir)
        #cwd = os.getcwd()
        #os.chdir(output_dir)
        
        # Barrier so that spawn is atomic between processes.
        # This is to make sure that vasp processes are spawned one by one according to
        # MPI policy (hopefully on adjacent nodes)
        # (might be MPI implementation dependent...)

        #for i in range(self.commsize):
        #    self.comm.Barrier()
        #    if i == self.commrank:
        failed_dir = []
        vasprundirs = self.comm.gather(output_dir,root=0)
        #print(self.commrank)
        if self.commrank == 0:
            start = timer()
            commspawn = [MPI.COMM_SELF.Spawn(self.path_to_vasp, #/home/issp/vasp/vasp.5.3.5/bin/vasp",
                                             args=[vasprundir,],
                                             maxprocs=self.nprocs) for vasprundir in vasprundirs]
            end = timer()
            print("rank ",self.worldrank," took ", end-start, " to spawn")
            sys.stdout.flush()
            start = timer()
            exitcode = np.array(0, dtype=np.intc)
            i = 0
            for comm in commspawn:
                comm.Bcast([exitcode, MPI.INT], root=0)
                comm.Disconnect()
                if exitcode != 0:
                    failed_dir.append(vasprundirs[i])
                i = i + 1
            end = timer()
            print("rank ", self.worldrank, " took ", end-start, " for vasp execution")

            if len(failed_dir) != 0:
                print("vasp failed in directories: \n "+"\n".join(failed_dir))
                sys.stdout.flush()
                if rerun == 0:
                    MPI.COMM_WORLD.Abort()
        self.comm.Barrier()

        # Rerun if VASP failed
        failed_dir = self.comm.bcast(failed_dir,root=0)
        if len(failed_dir) != 0 and rerun == 1:
            if self.commrank == 0:
                print("falling back to damped algorithm")
            poscar = Poscar.from_file(output_dir+"/CONTCAR")
            VaspInput.update({'POSCAR':poscar})
            incar = VaspInput.get('INCAR')
            incar.update({'IBRION':3,'POTIM':0.2})
            VaspInput.update({'INCAR':incar})
            rerun -= 1
            self.submit(VaspInput, output_dir, rerun)

        elif len(failed_dir) != 0 and rerun > 0:
            if self.commrank == 0:
                print("rerunning with copied CONTCARS")
            poscar = Poscar.from_file(output_dir+"/CONTCAR")
            VaspInput.update({'POSCAR':poscar})
            rerun -= 1
            self.submit(VaspInput, output_dir, rerun)
            

        
        #commspawn = MPI.COMM_SELF.Spawn(self.path_to_vasp, #/home/issp/vasp/vasp.5.3.5/bin/vasp",
        #                                args=[output_dir],
        #                                   maxprocs=self.nprocs)



        # Spawn is too slow, can't afford to make it atomic
        #commspawn = MPI.COMM_SELF.Spawn(self.path_to_vasp, #/home/issp/vasp/vasp.5.3.5/bin/vasp",
         #                               args=[output_dir,],
         #                               maxprocs=self.nprocs)
#        sendbuffer = create_string_buffer(output_dir.encode('utf-8'),255)
#        commspawn.Bcast([sendbuffer, 255, MPI.CHAR], root=MPI.ROOT)
        #commspawn.Barrier()
        #commspawn.Disconnect()
        #os.chdir(cwd)
        return 0
