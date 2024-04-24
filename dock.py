import os
import subprocess
import re
import time
import shutil
import pandas as pd

from typing import Dict, List, Union
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import meeko
    _meeko_available = True
except ImportError:
    _meeko_available = False


def delete_all():
    # DEBUGGING
    if os.path.exists('./configs/'):
        shutil.rmtree('./configs')
    if os.path.exists('./ligands/'):
        shutil.rmtree('./ligands/')
    if os.path.exists('./ligands_tmp/'):
        shutil.rmtree('./ligands_tmp/')
    if os.path.exists('./outputs_tmp/'):
        shutil.rmtree('./outputs_tmp/')
    if os.path.exists('./logs/'):
        shutil.rmtree('./logs/')
    if os.path.exists('./outputs/'):
        shutil.rmtree('./outputs/')

def move_files_from_dir(source_dir_path: str, dest_dir_path: str):
    files = os.listdir(source_dir_path)
    for file in files:
        source_file = os.path.join(source_dir_path, file)
        destination_file = os.path.join(dest_dir_path, file)
        shutil.move(source_file, destination_file)

def delete_dir_contents(dir_path: str):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


class ShellProcessExecutor:
    def __init__(self, print_output: bool = False, timeout_duration: int = 600) -> None:
        self.print_output = print_output
        self.timeout_duration = timeout_duration


class OBabelLigandPreparation(ShellProcessExecutor):
    """
        Using obabel command line to prepare ligand.
        Installation instructions at https://openbabel.org/docs/Installation/install.html
    """
    def __call__(self, smi: str, ligand_path: str) -> None:
        cmd_str = f'obabel -:"{smi}" -O "{ligand_path}" -h --gen3d'
        if not self.print_output:
            cmd_str += ' > /dev/null 2>&1'
        with subprocess.Popen(cmd_str, shell=True, start_new_session=True) as proc:
            try:
                if proc.wait(timeout=self.timeout_duration) != 0 and self.print_output:
                    print(f"Error preparing ligand: {smi}")
            except subprocess.TimeoutExpired:
                proc.kill()


class AutoDockLigandPreparation(ShellProcessExecutor):
    """
        Using AutoDock to prepare ligand (a python3 translated version).
        GitHub repository at https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
        Installation: "pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"
    """
    def __call__(self, smi: str, ligand_path: str) -> None:
        with subprocess.Popen(f'prepare_ligand4',
        shell=True, start_new_session=True) as proc:
            try:
                proc.wait(timeout=self.timeout_duration)
            except subprocess.TimeoutExpired:
                proc.kill()


class MeekoLigandPreparation(ShellProcessExecutor):
    """
        Using Meeko to prepare ligand.
        GitHub repository: https://github.com/forlilab/Meeko
        Installation: "pip install meeko"
    """
    def __init__(self, print_output: bool, timeout_duration: int = 600) -> None:
        super().__init__(print_output, timeout_duration)
        if not _meeko_available:
            raise Exception("Meeko package hasn't been installed.")

    def __call__(self, smi: str, ligand_path: str) -> None:
        mol = Chem.MolFromSmiles(smi)

        # Add hydrogen and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        w = Chem.SDWriter(f'{ligand_path}.sdf')
        w.write(mol)
        w.close()

        cmd_str = f'mk_prepare_ligand.py -i {ligand_path}.sdf -o {ligand_path}'
        if not self.print_output:
            cmd_str += ' > /dev/null 2>&1'
        with subprocess.Popen(cmd_str,
        shell=True, start_new_session=True) as proc:
            try:
                if proc.wait(timeout=self.timeout_duration) != 0 and self.print_output:
                    print(f"Error preparing ligand: {smi}")
            except subprocess.TimeoutExpired:
                proc.kill()
        
        if os.path.isfile(f'{ligand_path}.sdf'):
            os.remove(f'{ligand_path}.sdf')


class TimedProfiler:
    def __init__(self) -> None:
        self._count = 0
        self._total = 0
        self._average = 0
    
    def _add_value(self, value):
        self._total += value
        self._count += 1
        self._average = self._total / self._count
    
    def time_it(self, fn, *args, **kwargs):
        start_time = time.time()
        fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
    
    def get_average(self):
        return self._average


class VinaDockingModule:
    def __init__(self, 
                 vina_cmd: str,
                 receptor_pdbqt_file: str,
                 center_pos: List[float],
                 size: List[float],
                 ligand_dir_path: str = './ligands/',
                 output_dir_path: str = './outputs/',
                 log_dir_path: str = './logs/',
                 config_dir_path: str = './configs/',
                 tmp_ligand_dir_path: str = './ligands_tmp/',
                 tmp_output_dir_path: str = './outputs_tmp/',
                 tmp_config_file_path: str = './config_tmp.conf',
                 keep_ligand_file: bool = True,
                 keep_output_file: bool = True,
                 keep_log_file: bool = True,
                 keep_config_file: bool = True,
                 timeout_duration: int = 600,
                 additional_vina_args: Dict[str, str] = None,
                 ligand_preparation_fn: callable = OBabelLigandPreparation(),
                 print_msgs: bool = False,
                 print_vina_output: bool = False,
                 debug: bool = False) -> None:
        """
            Parameters:
            - vina_cmd: Command line prefix to execute vina command (e.g. ./qvina2.1)
            - receptor_pdbqt_file: Cleaned receptor PDBQT file to use for docking
            - center_pos: 3-dim list containing (x,y,z) coordinates of grid box
            - size: 3-dim list containing sizing information of grid box in (x,y,z) directions
            - ligand_dir_path: Path to save ligand preparation files
            - output_dir_path: Path to save docking output files
            - log_dir_path: Path to save log files
            - config_dir_path: Path to save config files
            - tmp_ligand_dir_path: Path to save temporary ligand files (for batched docking)
            - tmp_output_dir_path: Path to save temporary output files (for batched docking)
            - tmp_config_file_path: Path to save temporary config file (for batched docking)
            - keep_ligand_file: Save ligand file (True) or not (False)
            - keep_output_file: Save output file (True) or not (False)
            - keep_log_file: Save log file (True) or not (False)
            - keep_config_file: Save config file (True) or not (False)
            - timeout_duration: Timeout in seconds before new process automatically stops
            - additional_vina_args: Dictionary of additional Vina command arguments (e.g. {"cpu": "5"})
            - ligand_preparation_fn: Function to prepare molecule for docking. Should take the \
                argument format (smiles string, ligand_path, timeout_duration)
            - print_msgs: Show Python print messages in console (True) or not (False)
            - print_vina_output: Show Vina docking output in console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
        """

        # Check if certain docking aspects are supported (has vina command arguments e.g. '--log')
        try:
            with subprocess.Popen(f"{vina_cmd} --help_advanced",
                    stdout=subprocess.PIPE,
                    shell=True) as proc:
                proc.wait(timeout=timeout_duration)
                result = proc.stdout.read()
                if result is not None:
                    result = str(result)
                    self.batch_docking_support = False
                    self.logging_support = False

                    if "--log" in result:
                        self.logging_support = True
                    if "--ligand_directory" in result:
                        self.batch_docking_support = True
        except subprocess.TimeoutExpired:
            proc.kill()
            self.logging_support = False
            self.batch_docking_support = False
        
        if not os.path.isfile(receptor_pdbqt_file):
            raise Exception(rf'Protein file: {receptor_pdbqt_file} not found')
        
        if len(center_pos) != 3:
            raise Exception(f"center_pos must contain 3 values, {center_pos} was provided")

        if len(size) != 3:
            raise Exception(f"size must contain 3 values, {size} was provided")
        
        if print_msgs and not self.logging_support:
            print("Log files are not supported with this Vina command")
        
        if print_msgs and self.batch_docking_support:
            print("Batched docking is enabled, and logging is now disabled")
            self.logging_support = False
        elif print_msgs and not self.batch_docking_support:
            print("Batched docking is not supported with selected Vina version")

        self.vina_cmd = vina_cmd
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center_pos = center_pos
        self.size = size
        self.ligand_dir_path = ligand_dir_path
        self.output_dir_path = output_dir_path
        self.log_dir_path = log_dir_path
        self.config_dir_path = config_dir_path
        self.tmp_ligand_dir_path = tmp_ligand_dir_path
        self.tmp_output_dir_path = tmp_output_dir_path
        self.tmp_config_file_path = tmp_config_file_path
        self.keep_ligand_file = keep_ligand_file
        self.keep_output_file = keep_output_file
        self.keep_log_file = keep_log_file
        self.keep_config_file = keep_config_file
        self.timeout_duration = timeout_duration
        self.additional_vina_args = additional_vina_args
        self.ligand_preparation_fn = ligand_preparation_fn  # should follow format (smiles string, ligand_path)
        self.print_msgs = print_msgs
        self.print_vina_output = print_vina_output
        self.debug = debug

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()

        os.makedirs(output_dir_path, exist_ok=True)
        os.makedirs(config_dir_path, exist_ok=True)
        os.makedirs(ligand_dir_path, exist_ok=True)
        if self.logging_support:
            os.makedirs(log_dir_path, exist_ok=True)
        if self.batch_docking_support:
            os.makedirs(tmp_ligand_dir_path, exist_ok=True)
            os.makedirs(tmp_output_dir_path, exist_ok=True)
    
    def __call__(self, smi: Union[str, List[str]], redo_calculation: bool = False) -> List[Union[float, None]]:
        """
        Parameters:
        - smi: SMILES strings to perform docking. A single string activates single-ligand docking mode, while \
            multiple strings utilizes batched docking (if Vina version allows it).
        - redo_calculation: Force redo of ligand preparation, remake config file, and recalculate \
            binding affinity score.
        """
        binding_scores = []
        output_paths = []  # for batched docking

        if type(smi) is str:
            smi = [smi]

        for smi_str in smi:
            # sanitize SMILES string to save as file name
            ligand_name = smi_str.replace('(', '{').replace(')', '}').replace('\\', r'%5C').replace('/', r'%2F').replace('#', '%23')

            # Get proper directory/file pathing
            if self.batch_docking_support:
                tmp_ligand_path = self.tmp_ligand_dir_path + ligand_name + '.pdbqt'
                tmp_output_path = self.tmp_output_dir_path + ligand_name + '_out.pdbqt'

            ligand_path = self.ligand_dir_path + ligand_name + '.pdbqt'
            output_path = self.output_dir_path + ligand_name + '_out.pdbqt'
            config_path = self.config_dir_path + ligand_name + '.conf'
            output_exists = os.path.exists(output_path)  # docking already performed

            if self.logging_support:
                log_path = self.log_dir_path + ligand_name + '_log.txt'
            else:
                log_path = None

            if output_exists and not redo_calculation:
                output_paths.append(tmp_output_path)
                binding_scores.append(self._get_output_score(output_path))
                continue
            elif self.batch_docking_support:
                output_paths.append(tmp_output_path)
                binding_scores.append(None)

            # Prepare ligand before docking
            if os.path.isfile(ligand_path) and not redo_calculation and self.batch_docking_support:
                shutil.move(ligand_path, tmp_ligand_path)
                if self.print_msgs:
                    print(f'Ligand file: {ligand_path!r} already exists, moving to {tmp_ligand_path!r}')
            elif not os.path.isfile(ligand_path) or redo_calculation:
                if self.batch_docking_support:
                    save_ligand_path = tmp_ligand_path
                else:
                    save_ligand_path = ligand_path

                if self.debug:
                    self.preparation_profiler.time_it(self.ligand_preparation_fn, smi_str, save_ligand_path)
                else:
                    self.ligand_preparation_fn(smi_str, save_ligand_path)
            elif self.print_msgs:
                print(f'Ligand file: {ligand_path!r} already exists')
        
            # Make config file for non-batching
            if not self.batch_docking_support and (not os.path.isfile(output_path) or redo_calculation):
                conf = self._default_conf_str()
                conf += f'ligand = {ligand_path}\n'
                conf += f'out = {output_path}\n'

                with open(config_path, 'w') as f:
                    f.write(conf)
    
            # Perform docking procedure for non-batching
            if not self.batch_docking_support and (not os.path.isfile(output_path) or redo_calculation):
                if self.debug:
                    self.docking_profiler.time_it(self._run_vina, config_path, log_path)
                else:
                    self._run_vina(config_path, log_path)
            
            if not self.batch_docking_support:
                binding_scores.append(self._get_output_score(output_path))
        
                if not self.keep_ligand_file and os.path.exists(ligand_path):
                    os.remove(ligand_path)
                if self.logging_support and not self.keep_log_file and os.path.exists(log_path):
                    os.remove(log_path)
                if not self.keep_config_file and os.path.exists(config_path):
                    os.remove(config_path)
                if not self.keep_output_file and os.path.exists(output_path):
                    os.remove(output_path)
        
        if self.batch_docking_support:
            # Write config file
            conf = self._default_conf_str()
            conf += f'ligand_directory = {self.tmp_ligand_dir_path}\n'
            conf += f'output_directory = {self.tmp_output_dir_path}\n'

            with open(self.tmp_config_file_path, 'w') as f:
                f.write(conf)
            
            # Perform docking procedure
            if self.debug:
                self.docking_profiler.time_it(self._run_vina, self.tmp_config_file_path, None)
            else:
                self._run_vina(self.tmp_config_file_path, None)
            
            # Gather binding scores
            for i in range(len(binding_scores)):
                if binding_scores[i] is None:
                    binding_scores[i] = self._get_output_score(output_paths[i])
            
            # Move files from temporary to proper directory (or delete if redoing calculation)
            if self.keep_ligand_file:
                move_files_from_dir(self.tmp_ligand_dir_path, self.ligand_dir_path)
            else:
                delete_dir_contents(self.tmp_ligand_dir_path)

            if self.keep_output_file:
                move_files_from_dir(self.tmp_output_dir_path, self.output_dir_path)
            else:
                delete_dir_contents(self.tmp_output_dir_path)

            
            # Remove temporary config file
            if os.path.exists(self.tmp_config_file_path):
                os.remove(self.tmp_config_file_path)
        return binding_scores
    
    @staticmethod
    def _get_output_score(output_path: str) -> Union[float, None]:
        try:
            score = float("inf")
            with open(output_path, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
            return score
        except FileNotFoundError:
            return None
    
    def _default_conf_str(self):
        conf = f'receptor = {self.receptor_pdbqt_file}\n' + \
               f'center_x = {self.center_pos[0]}\n' + \
               f'center_y = {self.center_pos[1]}\n' + \
               f'center_z = {self.center_pos[2]}\n' + \
               f'size_x = {self.size[0]}\n' + \
               f'size_y = {self.size[1]}\n' + \
               f'size_z = {self.size[2]}\n'

        if self.additional_vina_args is not None:
                for k, v in self.additional_vina_args.items():
                    conf += f"{str(k)} = {str(v)} \n"
        return conf
    
    def _run_vina(self, config_path, log_path):
        try:
            cmd_str = f"{self.vina_cmd} --config {config_path}"
            if self.logging_support and log_path is not None:
                cmd_str += f" --log {log_path}"
            if not self.print_vina_output:
                cmd_str += " > /dev/null 2>&1"

            with subprocess.Popen(cmd_str, shell=True, start_new_session=True) as proc:
                proc.wait(timeout=self.timeout_duration)
        except subprocess.TimeoutExpired:
            proc.kill()
    
    def delete_tmp_files(self):
        if os.path.exists(self.tmp_config_file_path):
            os.remove(self.tmp_config_file_path)
        
        if os.path.exists(self.tmp_ligand_dir_path):
            shutil.rmtree(self.tmp_ligand_dir_path)
        
        if os.path.exists(self.tmp_output_dir_path):
            shutil.rmtree(self.tmp_output_dir_path)
        
    def delete_files(self, include_tmp_files: bool = True):
        if include_tmp_files:
            self.delete_tmp_files()
        
        if os.path.exists(self.ligand_dir_path):
            shutil.rmtree(self.ligand_dir_path)
        
        if os.path.exists(self.log_dir_path):
            shutil.rmtree(self.log_dir_path)
        
        if os.path.exists(self.config_dir_path):
            shutil.rmtree(self.config_dir_path)
        
        if os.path.exists(self.output_dir_path):
            shutil.rmtree(self.output_dir_path)



if __name__ == "__main__":
    data = pd.read_csv('zinc250k_first_1k.csv')
    timeout_duration = 600

    delete_all()

    preparation_fn = MeekoLigandPreparation(timeout_duration)

    docking_module = VinaDockingModule("./qvina2.1",
                                       "./advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       print_msgs=True,
                                       debug=True,
                                       ligand_preparation_fn=preparation_fn,
                                       timeout_duration=timeout_duration)
    print(docking_module(data['smiles'].iloc[:10]))

    delete_all()

    # QuickVina2-GPU 2.1
    docking_module = VinaDockingModule("/home/nick/Desktop/GPU-Docking/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1",
                                       "./advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       print_msgs=True,
                                       debug=True,
                                       ligand_preparation_fn=preparation_fn,
                                       timeout_duration=timeout_duration,
                                       additional_vina_args={"thread": "8000",
                                                             "opencl_binary_path" : "/home/nick/Desktop/GPU-Docking/Vina-GPU-2.1/QuickVina2-GPU-2.1"})
    print(docking_module(data['smiles'].iloc[:10]))
    exit(1)
    with tqdm(total=1000) as pbar:
        for smi in data.iloc[:1000]['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")
    

    # Vina-GPU+
    docking_module = VinaDockingModule("/home/nick/Desktop/GPU-Docking/Vina-GPU-2.0/Vina-GPU",
                                       "./advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       print_msgs=True,
                                       debug=True,
                                       ligand_preparation_fn=preparation_fn,
                                       timeout_duration=timeout_duration,
                                       additional_vina_args={"thread": "8000"})
    
    with tqdm(total=1000) as pbar:
        for smi in data['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")


    # AutoDock Vina-GPU 2.1
    docking_module = VinaDockingModule("/home/nick/Desktop/GPU-Docking/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1",
                                       "./advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       print_msgs=True,
                                       debug=True,
                                       ligand_preparation_fn=preparation_fn,
                                       timeout_duration=timeout_duration,
                                       additional_vina_args={"thread": "8000", 
                                                             "opencl_binary_path" : "/home/nick/Desktop/GPU-Docking/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"})
    
    with tqdm(total=1000) as pbar:
        for smi in data.iloc[:1000]['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")
    
    with open('autodock-vina-1.1.2.txt', 'w') as f:
        f.write(f"Lig. Prepare: {docking_module.preparation_profiler.get_average()}\n")
        f.write(f"Docking: {docking_module.docking_profiler.get_average()}")

    docking_module = VinaDockingModule("./autodock_vina_1_1_2/vina",
                                       "advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       print_msgs=True,
                                       debug=True)
    
    with tqdm(total=1000) as pbar:
        for smi in data.iloc[:1000]['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")
    
    with open('autodock-vina-1.1.2.txt', 'w') as f:
        f.write(f"Lig. Prepare: {docking_module.preparation_profiler.get_average()}\n")
        f.write(f"Docking: {docking_module.docking_profiler.get_average()}")

    docking_module = VinaDockingModule("./qvina2.1",
                                       "advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       debug=True)
    
    with tqdm(total=1000) as pbar:
        for smi in data.iloc[:1000]['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")
    
    with open('qvina2.1.txt', 'w') as f:
        f.write(f"Lig. Prepare: {docking_module.preparation_profiler.get_average()}\n")
        f.write(f"Docking: {docking_module.docking_profiler.get_average()}")
    

