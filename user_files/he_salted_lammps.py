#SALTED Initialization
#==============================================================================

from ase.io import lammpsdata
import time
import numpy as np

# Import SALTED functions
from salted import init_pred
from salted.sys_utils import ParseConfig
from salted.cp2k.utils import init_moments, init_ghost_integrals, compute_ghost_center
from salted.salted_prediction import build

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Get SALTED input 
inp = ParseConfig().parse_input()

# Load initialization structure and get general information 
structure = lammpsdata.read_lammps_data("he.lmp")
del structure[-1]
atomic_symbols = structure.get_chemical_symbols()
natoms = len(atomic_symbols)
cell = structure.get_cell()
bohr2angs = 0.529177210670
(saltedname, saltedpath, saltedtype,
    filename, species, average, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()


# Load information of pretrain SALTED model
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals = init_pred.build()

# Initialize integrals for computing the center of charge of the ghost density
kmin_integrals,kmin_harmonics = init_ghost_integrals(inp,cell/bohr2angs,lmax,nmax,species)

# Initialize calculation of density moments (only charge integrals will be needed in this example)
charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,0)
#==============================================================================

#lammps initialization
#==============================================================================
from lammps import lammps
import random
import os

os.makedirs("data", exist_ok=True)

# set the random seed
random.seed(80920231419)

# initiate main Lammps process (for the trajectory)
lmp_main = lammps()

## initialize the main Lammps process
lmp_main.file("_R_LAMMPS_IN_")
## initialize the temperature
lmp_main.command("velocity water create 300.0 %s dist gaussian" % ''.join(random.sample("0123456789",6)))

#read in run variable from lammps in for nsteps
#or read the control file for nsteps

#check set velocity or force on elec =0
#==============================================================================


nsteps = _R_NUMBER_OF_STEPS_
salted_timings = np.zeros(nsteps)
full_timings = np.zeros(nsteps)
for i in range(nsteps):
    #get time
    t_a = time.time()

    #run 1 step
    try:
        lmp_main.command("run 1")
    except Exception as e:
        print(f"LAMMPS crashed or failed at run step {i} with error: {e}")
        comm.Abort(1)

    #write the coordinates of the atoms
    lmp_main.command("write_data data/he_%s.lmp" % (i+1))

    #read in the conf for salted prediction
    structure = lammpsdata.read_lammps_data("data/he_%s.lmp" % (i+1))
    del structure[-1]

    #salted prediction
    start = time.time()
    coefs = build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,comm,size,rank,structure) 
    center_of_charge = compute_ghost_center(structure,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,kmin_integrals,kmin_harmonics,coefs)
    center_of_charge *= bohr2angs
    salted_timings[i] = time.time()-start

    #wrap
    for ik in range(3):
        center_of_charge[ik] = center_of_charge[ik] % cell[ik,ik] 

    #move the electron coordinate
    lmp_main.command("set atom 193 x %s y %s z %s" % (center_of_charge[0],center_of_charge[1],center_of_charge[2]))

    full_timings[i] = time.time()-t_a

lmp_main.command("write_restart _R_RESTART_OUT_")

f = open("timings.txt","w")
f.write("step full_time salted_time\n")
for i in range(nsteps):
    f.write(f"{i} {full_timings[i]} {salted_timings[i]}\n")
f.close()