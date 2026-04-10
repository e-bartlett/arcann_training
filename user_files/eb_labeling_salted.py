import json
import shutil
from salted import init_pred
from salted.sys_utils import ParseConfig
from salted.cp2k.utils import init_moments, init_ghost_integrals, compute_ghost_center
from salted.salted_prediction import build_orig
from ase.io import lammpsdata, read, write
from ase import Atom

## Load JSON data ../control/labeling_001.json
with open('../control/labeling_001.json', 'r') as f: 
    data = json.load(f)

candidates_count = data["systems_auto"]["he"]["candidates_count"]
print("candidates_count:", candidates_count)

is_checked = data["is_checked"]
if not is_checked:
    print("is_checked is False")
    print("Check labeling jobs before continuing")
    quit()

## Set up the SALTED stuff
shutil.copy('../user_files/inp.yaml', '.')
inp = ParseConfig().parse_input()

structure = lammpsdata.read_lammps_data("../user_files/he.lmp")
atomic_symbols = structure.get_chemical_symbols()[:-1]
natoms = len(atomic_symbols)
cell = structure.get_cell()
bohr2angs = 0.529177210670

(saltedname, saltedpath, saltedtype,
    filename, species, average, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut, zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, 
    HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals = init_pred.build()
kmin_integrals,kmin_harmonics = init_ghost_integrals(inp,cell/bohr2angs,lmax,nmax,species)
charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,0)

## For each of the candidates: predict the center of charge with SALTED
## Create updated labeling_XXXXX.xyz -> salted_labeling_XXXXX.xyz
## Create updated 2_labeling_00000-Forces.for -> 2_salted_labeling_00000-Forces.for
for i in range(candidates_count):
    iconf = str(i).zfill(5)
    print(f"Processing candidate {iconf}")

    # Calculate the center of charge
    structure = read(f"he/{iconf}/labeling_{iconf}.xyz")
    coefs = build_orig(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,structure)
    center_of_charge = compute_ghost_center(structure,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,kmin_integrals,kmin_harmonics,coefs)
    center_of_charge *= bohr2angs
    for ik in range(3):
        center_of_charge[ik] = center_of_charge[ik] % cell[ik,ik] 
    print("Center of charge:",center_of_charge)

    # Write coordinate file
    dummy_atom = Atom('X', position=center_of_charge)
    structure.append(dummy_atom)
    write(f"he/{iconf}/salted_labeling_{iconf}.xyz", structure)

    # Update forces file
    with open(f"he/{iconf}/2_labeling_{iconf}-Forces.for", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Sum" in line:
            lines.insert(i, " FORCES|    193 0.0 0.0 0.0 0.0\n")
            break 
    with open(f"he/{iconf}/2_salted_labeling_{iconf}-Forces.for", "w") as f:
        f.writelines(lines)

print("Done! Run labeling extract next.")