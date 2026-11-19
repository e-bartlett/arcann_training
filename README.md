<div style="text-align: center;">
<img src="./docs/arcann_logo.svg" alt="ArcaNN logo" style="width: 25%; height: auto;" />
</div>

---

[![GNU AGPL v3.0 License](https://img.shields.io/github/license/arcann-chem/arcann_training.svg)](https://github.com/arcann-chem/arcann_training/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD4DD00209A-004976.svg)](https://doi.org/10.1039/D4DD00209A)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.07751-b31b1b.svg)](https://doi.org/10.48550/arXiv.2407.07751)

[![Unit Tests Requirements](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml/badge.svg)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml)
[![Unit Tests Matrix](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml)
[![Docs](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml)

---

# Fork-Specific Installation

To check for updates, run `git fetch` and `git status`. To install the updates, run `git pull`.

## 1. ArcaNN environment

1. Clone the `share` branch of this fork:

   ```bash
   git clone -b share https://github.com/e-bartlett/arcann_training.git
   ```

2. Create a conda environment in `work`:

   ```bash
   conda create --prefix /kuhpc/work/thompson/t485j478/conda/arcann_mace
   ```

3. Activate it:

   ```bash
   conda activate /kuhpc/work/thompson/t485j478/conda/arcann_mace
   ```

4. Copy the list of conda packages:

   ```bash
   cp /kuhpc/scratch/thompson/e497b540/hydrated_electron/clean_electron_model/arcann_env/arcann_explicit.txt ./
   ```

5. Install the packages (with the environment activated):

   ```bash
   conda install --file arcann_explicit.txt
   ```

   > ⚠️ **This will take a while!**

6. From inside the `arcann_training` directory, install ArcaNN in editable mode:

   ```bash
   pip install -e .
   ```

   Because the install is editable (`-e`), you can run `git pull` in this clone to update ArcaNN.

7. Check the environment:

   ```bash
   conda activate <env name>
   python -c "import arcann_training, ase, MDAnalysis, dscribe, mpi4py; print(arcann_training.__file__)"
   ```

   Expected output (and nothing else):

   ```text
   /kuhpc/work/thompson/t485j478/fa26/arcann_mace/arcann_training/arcann_training/__init__.py
   ```

## 2. MACE environment

This environment, named `mace_lammps` is the env the MACE job scripts
run on A100 and A40 nodes. It includes mace-torch 0.3.16, torch 2.14.0+cu126, cuEquivariance 0.11.1 and a LAMMPS
Kokkos + ML-IAP build (with the EXTRA-DUMP package, for `dump dcd`).

1. Copy the environment files:

   ```bash
   cp /kuhpc/scratch/thompson/e497b540/hydrated_electron/clean_electron_model/mace_env/{mace.txt,mace_pip_requirements.txt,build_lammps_mace.sh} ./
   ```

2. Create the environment from the conda package list:

   ```bash
   conda create --prefix /kuhpc/work/thompson/t485j478/conda/mace_lammps --file mace.txt
   ```

3. Activate the environment:

   ```bash
   conda activate /kuhpc/work/thompson/t485j478/conda/mace_lammps
   export PYTHONNOUSERSITE=1
   ```

4. Install the pip requirements (you may want to submit this as a job or run on an interactive node):

   ```bash
   pip install -r mace_pip_requirements.txt
   ```

   > ⚠️ **This will take a while!**

5. Check the environment:

   ```bash
   python -c "import torch, mace, cuequivariance_torch, e3nn, ase; print(torch.__version__, torch.version.cuda, cuequivariance_torch.__version__)"
   ```

   Expected output:

   ```text
   2.14.0+cu126 12.6 0.11.1
   ```

   On a login node, an `Error while loading libcue_ops.so: libcuda.so.1 ...` line prints first.
   That's expected, because login nodes have no GPU driver.

6. Build LAMMPS (Kokkos + ML-IAP + EXTRA-DUMP) and install its Python module and the `lmp` executable into the
   environment. Run this on an A100 node from a scratch directory, because the build tree is large:

   ```bash
   srun -p bigjay --constraint=a100 --gres=gpu:1 -N1 -n1 -c16 -t2:00:00
   CONDA_ENV=/kuhpc/work/thompson/t485j478/conda/mace_lammps ./build_lammps_mace.sh
   ```

   > ⚠️ **This will take a while!** It ends with `BUILD OK`. The check before that should print
   > `has dump dcd: yes`.

   To check the lammps executable, run:

   ```bash
   module purge
   module load compiler/gcc/11.5 cuda/12.8 openmpi-cuda/5.0
   conda activate /kuhpc/work/thompson/t485j478/conda/mace_lammps
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
   export PYTHONHOME="$CONDA_PREFIX"   # only if needed
   lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps
   ```

7. The MACE job scripts in `user_files/` default to the env name `mace_lammps`
   (`MACE_ENV="${MACE_ENV:-mace_lammps}"`). Because this env was created with `--prefix`, point
   them at its full path in the shell you run ArcaNN from. SLURM passes it on to the jobs:

   ```bash
   export MACE_ENV=/kuhpc/work/thompson/t485j478/conda/mace_lammps
   ```

## 3. Prepare for running ArcaNN

When you are ready to convert your CP2K data to MACE compatible data,
you can use these codes from the Github repo as an example: 

- `erb_user_files/mace_prep.py`. You'll have to modify it to fit your directory structure and you'll have to remove the hydrated electron specific parts.

- `erb_user_files/tj_mace_prep.py`. This is how Thanuja modified my code to prepare his data.

## 4. Prepare your user files. 

The following guide lists each file in `user_files/` and what to change for a new system. It is based on how `erb_user_files` (hydrated electron in water) was adapted for a silica/water interface (Si, O, H; 864 atoms). Examples read *ERB* → *TJ*. Replace `<system>` with the name you put in `systems_auto`.

## Structure and MD inputs

- **`<system>.lmp`** (was `he.lmp`)
  - Replace with a LAMMPS data file for your system.
  - The atom-type order must match `properties.txt`, `pair_coeff` and `type_map`.
- **`<system>.pdb`** (was `he.pdb; not required but helpful for visualization.)
  - Replace with the same starting structure in PDB format.
- **`<system>.in`** (was `he.in`)
  - Unless you are running hydrated-electron mode, start from `mace_non_he/SYSTEM.in`, not `he.in`. Copy it to the top level of `user_files/` as `<system>.in`. ArcaNN only reads the top-level copy.
  - Keep `pair_style mliap unified _R_MACE_MODEL_ 0`. `exploration prepare` fails if `_R_MACE_MODEL_` is missing.
  - Set `pair_coeff * * <elements in type order>` (O H → Si O H).
  - Set groups and fixes for your system (as desired). For example, TJ holds atoms 107, 119, 131, 138 and 139 fixed with `setforce 0.0 0.0 0.0`. It applies NVE, CSVR and momentum fixes to the rest.
  - Adjust neighbor settings if needed (`neighbor 1.0` → `2.0`; `neigh_modify every 10` → `every 1`).
  - Keep every `_R_..._` placeholder, plus the `run _R_NUMBER_OF_STEPS_` and `write_restart _R_RESTART_OUT_` lines.
- **`<system>_mace_lammps.py`** (was `he_mace_lammps.py`)
  - Only for hydrated-electron mode.&#32;

## CP2K labeling inputs

- **`1_<system>_labeling_XXXXX_kuhpc.inp`** and **`2_<system>_labeling_XXXXX_kuhpc.inp`** (were `1_he_…` and `2_he_…`)
  - Rename both with your system name.
  - Modify the contents to perform single point calculations at the desired level of theory.
- **The step-2 file must keep two things,** whatever else changes:
  - `@SET SYSNAME 2_labeling__R_PADDEDSTEP_`
  - The `&PRINT` block with `PROGRAM_RUN_INFO` and `FORCES`. `labeling extract` reads `2_labeling_<step>-Force_Eval.fe` and `2_labeling_<step>-Forces.for`.&#32;

## Config and training files

- **`default_config.json`  is just an example containing the input.json fields for each phase, not strictly necessary.**
  - `systems_auto`: `[""]` → `["<system>"]`
  - `mlip_engine`: `deepmd` → `mace`
  - `mace_env`: path to your MACE conda environment
- **`properties.txt`**
  - List each element with its type index, in LAMMPS type order (ERB: O 1, H 2, X 3 → TJ: Si 1, O 2, H 3).
  - List each mass.&#32;
- **`mace_train.yaml`**
  - `E0s`: one entry per atomic number, as isolated-atom energies in eV at your DFT level (`{1: -13.659544, 8: -432.289058, 14: -102.898223}`).
  - Model size: interface shrank it for 864 atoms (`r_max` 6.0 → 5.0, `hidden_irreps` 128x0e + 128x1o → 64x0e + 64x1o, `max_ell` 3 → 2). You can test this and choose what you want.
  - `valid_batch_size`: you should test this and use as much as can fit in memory.
- **`dptrain_3.0.json`** (DeePMD only)
  - `type_map` (O, H, X → Si, O, H), `sel` (maximum neighbors per type), `neuron`, and `atom_ener` per type.
- **`machine.json`**
  - `project_name`: your username or account, in all 5 places (e497b540 → t485j478).
- **`mace_prep.py`** (converts the initial CP2K dataset to extxyz)
  - Rewrite it for your data: input folder, atom count, `symbols` in file order, cell, the config indices to read, CP2K output filenames and output name.
  - Remove the electron-centroid part at the end.
  - TJ's version is saved as `erb_user_files/tj_mace_prep.py`.&#32;

## Job scripts

- **`job_lammps-deepmd_explore_gpu_kuhpc.sh`** and **`job-array_lammps-deepmd_explore_gpu_kuhpc.s``h`**  **IMPORTANT**! These run mace even though the file name says deepmd.&#32;
  - Unless you are running hydrated-electron mode, copy the versions in `mace_non_he/` to the top level of `user_files/`. ArcaNN only reads the top-level copies.
  - The top-level erb versions run the hydrated-electron Python driver (`MACE_DRIVER_DIR`, centroid model).
  - Set the `MACE_ENV` default to your environment (`mace_lammps` → `/kuhpc/work/thompson/t485j478/conda/mace_lammps`).
  - Keep the file names. ArcaNN looks for `job_lammps-deepmd_explore_*` even when the engine is MACE.
- **`job_mace_train_gpu_kuhpc.sh`**
  - `MACE_ENV`: path to your environment.
  - Optional: add `--enable_cueq TRUE` if your environment has cuEquivariance (recommended and enabled in the latest mace\_lammps environment build).
- **`job_mace_freeze_gpu_kuhpc.sh`**
  - `MACE_ENV`: your environment.
  - `#SBATCH --mem`: test this for your system.

## No changes needed

- `job_CP2K_label_cpu_kuhpc.sh`, `job-array_CP2K_label_cpu_kuhpc.sh`
- `collect_traj.sh`, `install_deepmd-kit_env.sh`, `loss.py`
- `plot_dev.py`, `plot_loss.py`, `plot_traj_min_dev.py`

## Hydrated electron, DeePMD, or reference only (can delete for other systems)

- **Electron centroid model:** `model.py`, `readout.py`, `train_centroid.py`, `evaluate_centroid.py`, `job_mace_centroid_gpu_kuhpc.sh`, `compare_centroid_models.py`, `job_mace_centroid_compare_gpu_kuhpc.sh`
- **Electron data tools:** `centroid_label_from_cube.py`, `to_extxyz_centroid.py`, `analyze_dataset.py`, `plumed_he.dat`, `inp.yaml`
- `deepmd_npy_to_extxyz.py`,
- **DeePMD only:** `job_deepmd_train_gpu_kuhpc.sh`, `job_deepmd_freeze_gpu_kuhpc.sh`, `job_deepmd_compress_gpu_kuhpc.sh`, `job_deepmd_test_gpu_kuhpc.sh`, `dptrain_2.2.json`
- **Other:** `mace_default_input.json`, `tj_mace_prep.py` (reference only), `__pycache__/`


## 5. Default ArcaNN README

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
In its current version, it aims to simplify and to automate the iterative training process of a [DeePMD-kit](https://doi.org/10.1063/5.0155600) neural network potential for a user-chosen system, but the core concepts of the training procedure could be extended to other network architectures.
The main advantages of this code are its modularity, the ability to finely tune the training process to adapt to your system and workflow, and great traceability, as the code records every parameter set during the procedure.
During the iterative training process, you will iteratively train neural network potentials, use them as reactive force fields for molecular dynamics simulations (to explore the phase space), select and label some configurations based on a query by committee approach, and then train neural network potentials again with an improved training set, and so forth.
This workflow, sometimes referred to as active or concurrent learning, was heavily inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206), and we use their naming scheme for the steps of the iterative procedure.

We refer the reader to the [documentation](https://arcann-chem.github.io/arcann_training/) and the accompanying paper [ArcaNN: automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials](https://doi.org/10.1039/D4DD00209A).

## Installation ##

To install ArcaNN, please read the [documentation](https://arcann-chem.github.io/arcann_training/getting-started/requirements/).

## License ##

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

## How to cite ##

If you use this code, please cite the following publication:

David, R.; de la Puente, M.; Gomez, A.; Anton, O.; Stirnemann, G.; Laage, D. ArcaNN: automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials. Digital Discovery, 2024, DOI: [10.1039/D4DD00209A](https://doi.org/10.1039/D4DD00209A).

## Fundings & HPC Allocations ##

- Idex ANR-10-IDEX-0001-02PSL
- ERC Grant Agreement No. 757111
- GENCI Grant 2023-A0130707156

## Acknowledgments & Sources ##

- [Stackoverflow](https://stackoverflow.com/)

### Beta-testers ###

- Olaia Anton, Zakarya Benayad, Miguel de la Puente, Axel Gomez
- Oscar Gayraud, Pierre Girard, Anne Milet
- Meritxell Malagarriga Perez, Adrián García
- Ashley Borkowski, Pauf Neupane, Ward Thompson
- Hadi Dinpajooh

### Atomsk ###

- Hirel, P. Atomsk: A Tool for Manipulating and Converting Atomic Data Files. Comput. Phys. Commun. 2015, 197, 212–219. [https://doi.org/10.1016/j.cpc.2015.07.012](https://doi.org/10.1016/j.cpc.2015.07.012).

### VMD ###

- Humphrey, W.; Dalke, A.; Schulten, K. VMD: Visual Molecular Dynamics. J. Mol. Graph. 1996, 14 (1), 33–38. [https://doi.org/10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5).

### DeePMD-kit ###

- Zeng, J.; Zhang, D.; Lu, D.; Mo, P.; Li, Z.; Chen, Y.; Rynik, M.; Huang, L.; Li, Z.; Shi, S.; Wang, Y.; Ye, H.; Tuo, P.; Yang, J.; Ding, Y.; Li, Y.; Tisi, D.; Zeng, Q.; Bao, H.; Xia, Y.; Huang, J.; Muraoka, K.; Wang, Y.; Chang, J.; Yuan, F.; Bore, S. L.; Cai, C.; Lin, Y.; Wang, B.; Xu, J.; Zhu, J.-X.; Luo, C.; Zhang, Y.; Goodall, R. E. A.; Liang, W.; Singh, A. K.; Yao, S.; Zhang, J.; Wentzcovitch, R.; Han, J.; Liu, J.; Jia, W.; York, D. M.; E, W.; Car, R.; Zhang, L.; Wang, H. DeePMD-Kit v2: A Software Package for Deep Potential Models. J. Chem. Phys. 2023, 159 (5), 054801. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1063/5.0155600).
- Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-Kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics. Comput. Phys. Commun. 2018, 228, 178–184. [https://doi.org/10.1016/j.cpc.2018.03.016](https://doi.org/10.1016/j.cpc.2018.03.016).

### DP-Compress ###

- Lu, D.; Jiang, W.; Chen, Y.; Zhang, L.; Jia, W.; Wang, H.; Chen, M. DP Compress: A Model Compression Scheme for Generating Efficient Deep Potential Models. J. Chem. Theory Comput. 2022, 18 (9), 5559–5567. [https://doi.org/10.1021/acs.jctc.2c00102](https://doi.org/10.1021/acs.jctc.2c00102).

### Concurrent Learning ###

- Zhang, L.; Lin, D.-Y.; Wang, H.; Car, R.; E, W. Active Learning of Uniformly Accurate Interatomic Potentials for Materials Simulation. Phys. Rev. Materials 2019, 3 (2), 023804. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1103/PhysRevMaterials.3.023804)
- Zhang, Y.; Wang, H.; Chen, W.; Zeng, J.; Zhang, L.; Wang, H.; E, W. DP-GEN: A Concurrent Learning Platform for the Generation of Reliable Deep Learning Based Potential Energy Models. Comput. Phys. Commun. 2020, 253, 107206. [https://doi.org/10.1016/j.cpc.2020.107206](https://doi.org/10.1016/j.cpc.2020.107206).

### LAMMPS ###

- Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.; Crozier, P. S.; In ’T Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D.; Shan, R.; Stevens, M. J.; Tranchida, J.; Trott, C.; Plimpton, S. J. LAMMPS - a Flexible Simulation Tool for Particle-Based Materials Modeling at the Atomic, Meso, and Continuum Scales. Comput. Phys. Commun. 2022, 271, 108171. [https://doi.org/10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).

### i-PI ###

- Kapil, V.; Rossi, M.; Marsalek, O.; Petraglia, R.; Litman, Y.; Spura, T.; Cheng, B.; Cuzzocrea, A.; Meißner, R. H.; Wilkins, D. M.; Helfrecht, B. A.; Juda, P.; Bienvenue, S. P.; Fang, W.; Kessler, J.; Poltavsky, I.; Vandenbrande, S.; Wieme, J.; Corminboeuf, C.; Kühne, T. D.; Manolopoulos, D. E.; Markland, T. E.; Richardson, J. O.; Tkatchenko, A.; Tribello, G. A.; Van Speybroeck, V.; Ceriotti, M. I-PI 2.0: A Universal Force Engine for Advanced Molecular Simulations. Comput. Phys. Commun. 2019, 236, 214–223. [https://doi.org/10.1016/j.cpc.2018.09.020](https://doi.org/10.1016/j.cpc.2018.09.020).

### CP2K ###

- Kühne, T. D.; Iannuzzi, M.; Del Ben, M.; Rybkin, V. V.; Seewald, P.; Stein, F.; Laino, T.; Khaliullin, R. Z.; Schütt, O.; Schiffmann, F.; Golze, D.; Wilhelm, J.; Chulkov, S.; Bani-Hashemian, M. H.; Weber, V.; Borštnik, U.; Taillefumier, M.; Jakobovits, A. S.; Lazzaro, A.; Pabst, H.; Müller, T.; Schade, R.; Guidon, M.; Andermatt, S.; Holmberg, N.; Schenter, G. K.; Hehn, A.; Bussy, A.; Belleflamme, F.; Tabacchi, G.; Glöß, A.; Lass, M.; Bethune, I.; Mundy, C. J.; Plessl, C.; Watkins, M.; VandeVondele, J.; Krack, M.; Hutter, J. CP2K: An Electronic Structure and Molecular Dynamics Software Package - Quickstep: Efficient and Accurate Electronic Structure Calculations. J. Chem. Phys. 2020, 152 (19), 194103. [https://doi.org/10.1063/5.0007045](https://doi.org/10.1063/5.0007045).

