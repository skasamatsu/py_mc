import numpy as np
from pymatgen import Structure
from pymatgen.io.vasp import Poscar, Kpoints, Potcar, VaspInput
import subprocess, os
import multiprocess as mp

class vasp_run:
    # Single vasp run (no parallel replicas)
    def __init__(self, vasp_run_cmd):
        self.vasp_run_cmd = vasp_run_cmd
    def submit(self, VaspInput, output_dir):
        VaspInput.write_input(output_dir=output_dir)
        cwd = os.getcwd()
        os.chdir(output_dir)
        stdout = open("stdout.log", "w")
        stderr = open("stderr.log", "w")
        stdin = open(os.devnull, "r")
        p = subprocess.Popen(vasp_run_cmd, stdout=stdout, stderr=stderr, stdin=stdin, shell=True)
        os.chdir(cwd)
        exitcode = p.wait()
        return exitcode

def submit_bulkjob(vaspruns, path_to_vasp, n_mpiprocs, n_ompthreads):
    joblist = open("joblist.txt", "w")
    if n_ompthreads =! 0:
        progtype = "H"+str(n_ompthreads)
    else:
        progtype = "M"
    for vasprun in vaspruns:
        vasprun[0].write_input(output_dir=vasprun[1])
        joblist.write(
            ";".join([path_to_vasp, str(n_mpiprocs), progtype, vasprun[1]]) + "\n")
    stdout = open("stdout.log", "w")
    stderr = open("stderr.log", "w")
    stdin = open(os.devnull, "r")
    p = subprocess.Popen("bulkjob ./joblist.txt", stdout=stdout, stderr=stderr, stdin=stdin,
                         shell=True)
    exitcode = p.wait()
    return exitcode
    
def vasp_bulkjob_qwatcher(q, path_to_vasp, nvaspruns, n_replicas, n_mpiprocs, n_ompthreads):
    # Meant to be run as a separate process that
    # receives vasp jobs, writes a joblist file, and
    # submits a bulkjob in ISSP System B (SGI ICE XA).

    nvaspruns_in = nvaspruns
    n_replicas_in = n_replicas
    while True:
        vaspruns = []
        for i in range(nvaspruns):
            vasprun = q.get(timeout=1800)
            if vasprun == "terminate":
                n_replicas -= 1
                q.task_done()
                if n_replicas < nvaspruns:
                    nvaspruns = n_replicas
                    break
            vaspruns.append(vasprun)
        if n_replicas == 0:
            # We're done for all replicas; Reset
            nvaspruns = nvaspruns_in
            n_replicas = n_replicas_in
        else:
            # We're not done yet; submit bulkjob
            exitcode = submit_bulkjob(
                vaspruns, path_to_vasp, n_mpiprocs, n_ompthreads)
            for vasprun in vaspruns:
                # vasprun[2] is the pipe end to corresponding replica
                vasprun[2].send(exitcode)
                vasprun[2].close()
            
class vasp_run_use_queue:
    def __init__(self, queue):
        self.queue = queue
        
    def submit(self, VaspInput, output_dir):
        # submit job to qwatcher
        # send pipe to get job status
        my_pipe_end, qwatcher_pipe_end = mp.Pipe()
        self.queue.put([VaspInput, output_dir, qwatcher_pipe_end])
        exitcode = my_pipe_end.recv()
        return exitcode

    def terminate(self):
        self.queue.put("terminate")

if __name__ == "__main__":
    vasp_run_cmd = "mpijob /home/issp/vasp/vasp.5.3.5/bin/vasp.gamma"
    vaspinput = VaspInput.from_directory('baseinput')
    p = vasp_run(vaspinput, 'output', vasp_run_cmd)
    print "vasp exited with exit code", p.wait()


    
    
    
