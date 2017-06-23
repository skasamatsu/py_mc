import numpy as np
from pymatgen import Structure
from pymatgen.io.vasp import Poscar, Kpoints, Potcar, VaspInput
import subprocess, os


def vasp_run(VaspInput, output_dir, vasp_run_cmd):
    VaspInput.write_input(output_dir=output_dir)
    cwd = os.getcwd()
    os.chdir(output_dir)
    stdout = open("stdout.log", "w")
    stderr = open("stderr.log", "w")
    stdin = open(os.devnull, "r")
    p = subprocess.Popen(vasp_run_cmd, stdout=stdout, stderr=stderr, stdin=stdin, shell=True)
    os.chdir(cwd)
    return p

def vasp_bulkjob_server(q, nvaspruns, n_mpiprocs, n_ompthreads):
    # Meant to be run as a separate process that
    # receives vasp jobs, writes a joblist file, and
    # submits a bulkjob in ISSP System B (SGI ICE XA)
    while True:
        vaspruns = []
        for i in range(nvaspruns):
            vasprun = q.get(timeout=1800)
            if vasprun == "terminate":
                break
            vaspruns.append(vasprun)
        
                


if __name__ == "__main__":
    vasp_run_cmd = "mpijob /home/issp/vasp/vasp.5.3.5/bin/vasp.gamma"
    vaspinput = VaspInput.from_directory('baseinput')
    p = vasp_run(vaspinput, 'output', vasp_run_cmd)
    print "vasp exited with exit code", p.wait()


    
    
    
