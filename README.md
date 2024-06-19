# PyVina
A Python Module implemented for using the Vina suite of molecular docking software.

Tested using [QuickVina 2.1](https://qvina.github.io/), [AutoDock Vina](https://vina.scripps.edu/), [QuickVina2 GPU](https://github.com/DeltaGroupNJUPT/QuickVina2-GPU), [Vina GPU 2](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0), [QuickVina2 GPU 2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1), and [Vina GPU 2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1).

## Installation
- open terminal in project directory
- git clone https://github.com/Pixelatory/PyVina/
- pip install meeko (for MeekoLigandPreparator)

- Note: installation of Vina docking program is done separately.

## Example Usage: QuickVina2 GPU 2.1 with CPU fallback
```
from PyVina.ligand_preparators import MeekoLigandPreparator
from PyVina.dock import VinaDocking
from PyVina.exceptions import OutOfGPUMemoryError
import os

ligand_preparator = MeekoLigandPreparator(print_output=False,
                                          timeout_duration=None,
                                          n_workers=os.cpu_count())

docking_module_gpu = VinaDocking("path/to/QuickVina2-GPU-2-1",
                                 "path/to/cleaned/receptor/pdbqt",
                                 [7.750, -14.556, 6.747],
                                 [20, 20, 20],
                                 keep_ligand_file=True,
                                 keep_config_file=True,
                                 keep_log_file=True,
                                 keep_output_file=True,
                                 ligand_preparation_fn=ligand_preparator,
                                 timeout_duration=None,
                                 gpu_ids=[0],
                                 additional_vina_args={"thread": "5000",
                                                       "opencl_binary_path": "path/to/QuickVina2-GPU-2.1"})

docking_module_cpu = VinaDocking("path/to/qvina2.1",
                                 "path/to/cleaned/receptor/pdbqt",
                                 [7.750, -14.556, 6.747],
                                 [20, 20, 20],
                                 keep_ligand_file=True,
                                 keep_config_file=True,
                                 keep_log_file=True,
                                 keep_output_file=True,
                                 ligand_preparation_fn=ligand_preparator,
                                 timeout_duration=None,
                                 gpu_ids=None,
                                 additional_vina_args={"cpu": os.cpu_count()})

try:
  print(docking_module_gpu(["CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]))
except OutOfGPUMemoryError:
  print(docking_module_cpu(["CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]))

docking_module_cpu.delete_files()
docking_module_gpu.delete_files()

```
