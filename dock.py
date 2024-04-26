import os
import subprocess
import re
import time
import shutil
import multiprocessing

from typing import Dict, Iterable, List, Union
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from functools import partial

try:
    import meeko
    _meeko_available = True
except ImportError:
    _meeko_available = False


def delete_all():
    # DEBUGGING PURPOSES
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

def sanitize_smi_name_for_file(smi: str):
    return smi.replace('(', '{').replace(')', '}').replace('\\', r'%5C').replace('/', r'%2F').replace('#', '%23')

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

def execute_shell_process(cmd_str: str, error_msg: str = None, print_output: bool = False, timeout_duration: int = None) -> None:
    with subprocess.Popen(cmd_str, shell=True, start_new_session=True) as proc:
        try:
            if proc.wait(timeout=timeout_duration) != 0 \
                and print_output \
                and error_msg is not None:
                print(error_msg)
        except subprocess.TimeoutExpired:
            proc.kill()


class ShellProcessExecutor:
    def __init__(self, print_output: bool = False, timeout_duration: int = 600) -> None:
        self.print_output = print_output
        self.timeout_duration = timeout_duration
    
    def _execute_shell_process(self, cmd_str: str, error_msg: str = None):
        with subprocess.Popen(cmd_str, shell=True, start_new_session=True) as proc:
            try:
                if proc.wait(timeout=self.timeout_duration) != 0 \
                    and self.print_output \
                    and error_msg is not None:
                    print(error_msg)
            except subprocess.TimeoutExpired:
                proc.kill()


class PooledWorkerExecutor:
    def __init__(self, num_workers=None, timeout_duration: int = 600):
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(num_workers)
        self.timeout_duration = timeout_duration

    def map(self, fn, iterable):
        return self.pool.map_async(fn, iterable)

    def starmap(self, fn, iterable):
        return self.pool.starmap_async(fn, iterable)

    def apply(self, fn):
        return self.pool.apply_async(fn)

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()


class OBabelLigandPreparator(ShellProcessExecutor, PooledWorkerExecutor):
    """
        Using obabel command line to prepare ligand.
        Installation instructions at https://openbabel.org/docs/Installation/install.html
    """
    def __call__(self, smi: str, ligand_path: str) -> None:
        cmd_str = f'obabel -:"{smi}" -O "{ligand_path}" -h --gen3d'
        if not self.print_output:
            cmd_str += ' > /dev/null 2>&1'
        self._execute_shell_process(cmd_str, f"Error preparing ligand: {smi}")


class AutoDockLigandPreparator(ShellProcessExecutor):
    """
        Using AutoDock to prepare ligand (a python3 translated version).
        GitHub repository at https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
        Installation: "pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"
    """
    def __call__(self, smi: str, ligand_path: str) -> None:
        cmd_str = f'prepare_ligand4'
        self._execute_shell_process(cmd_str, f"Error preparing ligand: {smi}")


class MeekoLigandPreparator(ShellProcessExecutor):
    """
        Using Meeko to prepare ligand.
        GitHub repository: https://github.com/forlilab/Meeko
        Installation: "pip install meeko"
    """
    def __init__(self, print_output: bool, timeout_duration: int = None, n_workers: Union[int, None] = -1) -> None:
        super().__init__(print_output, timeout_duration)
        if not _meeko_available:
            raise Exception("Meeko package hasn't been installed.")
        
        if n_workers != -1:
            self.pool_executor = PooledWorkerExecutor(n_workers, timeout_duration)
        else:
            self.pool_executor = None

    def __call__(self, smi: Union[str, List[str]], ligand_path: Union[str, List[str]]) -> List[bool]:
        """
            returns: List containing whether new ligand file was created (True) or is already existing (False)
        """
        if smi is type(str):
            smi = [smi]
        if ligand_path is type(str):
            ligand_path = [ligand_path]
        
        if self.pool_executor is not None:
            tmp = partial(self._prepare_ligand, print_output=self.print_output)
            res = self.pool_executor.starmap(tmp, zip(smi, ligand_path))
            return res.get()
        else:
            res = []
            for i in range(len(smi)):
                res.append(self._prepare_ligand(smi[i], ligand_path[i], self.print_output))
            return res
    
    @staticmethod
    def _prepare_ligand(smi: str, ligand_path: str, print_output: bool = False) -> bool:
        """
            Note: Potentially unsafe for concurrent application with batched docking.
            Ensure unique ligands for preparation.

            returns: True if new ligand file was produced, False if ligand file already exists
        """
        if not os.path.exists(ligand_path):
            mol = Chem.MolFromSmiles(smi)

            # Add hydrogen and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            w = Chem.SDWriter(f'{ligand_path}.sdf')
            w.write(mol)
            w.close()

            cmd_str = f'mk_prepare_ligand.py -i {ligand_path}.sdf -o {ligand_path}'
            if not print_output:
                cmd_str += ' > /dev/null 2>&1'
            
            execute_shell_process(cmd_str, f"Error preparing ligand: {smi}")
            
            if os.path.isfile(f'{ligand_path}.sdf'):
                os.remove(f'{ligand_path}.sdf')
            
            return True
        return False


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
        res = fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
        return res
    
    def get_average(self):
        return self._average


class VinaDockingModule:
    def __init__(self, 
                 vina_cmd: str,
                 receptor_pdbqt_file: str,
                 center_pos: List[float],
                 size: List[float],
                 ligand_dir_path: str = 'ligands/',
                 output_dir_path: str = 'outputs/',
                 log_dir_path: str = 'logs/',
                 config_dir_path: str = 'configs/',
                 tmp_ligand_dir_path: str = 'ligands_tmp/',
                 tmp_output_dir_path: str = 'outputs_tmp/',
                 tmp_config_file_path: str = 'config_tmp.conf',
                 keep_ligand_file: bool = True,
                 keep_output_file: bool = True,
                 keep_log_file: bool = True,
                 keep_config_file: bool = True,
                 timeout_duration: int = 1000,
                 additional_vina_args: Dict[str, str] = {},
                 ligand_preparation_fn: callable = OBabelLigandPreparator(),
                 vina_cwd: str = None,
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
            - ligand_preparation_fn: Function/Class callable to prepare molecule for docking. Should take the \
                argument format (smiles strings, ligand paths)
            - vina_cwd: Change current working directory of Vina shell (sometimes needed for GPU versions \
                and incorrect openCL pathing)
            - print_msgs: Show Python print messages in console (True) or not (False)
            - print_vina_output: Show Vina docking output in console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
        """

        # Check if certain docking aspects are supported (has vina command arguments e.g. '--log')
        try:
            with subprocess.Popen(f"{vina_cmd} --help_advanced",
                    stdout=subprocess.PIPE,
                    shell=True) as proc:
                if proc.wait(timeout=timeout_duration) == 0:
                    result = proc.stdout.read()
                    if result is not None:
                        result = str(result)
                        self.batch_docking_support = False
                        self.logging_support = False

                        if "--log" in result:
                            self.logging_support = True
                        if "--ligand_directory" in result:
                            self.batch_docking_support = True
                else:
                    raise Exception(f"Vina command '{vina_cmd}' returned unsuccessfully.")
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
            print("Batched docking is enabled; log and config files will not be saved")
            self.logging_support = False
        elif print_msgs and not self.batch_docking_support:
            print("Batched docking is disabled; not supported with selected Vina version")

        self.vina_cmd = vina_cmd
        self.receptor_pdbqt_file = os.path.abspath(receptor_pdbqt_file)
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
        self.vina_cwd = vina_cwd
        self.print_msgs = print_msgs
        self.print_vina_output = print_vina_output
        self.debug = debug

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()
        
        # TODO: may have to change current working directory (os.chdir()) back and forth depending on Vina version: use argument for that.

        os.makedirs(output_dir_path, exist_ok=True)
        os.makedirs(ligand_dir_path, exist_ok=True)
        if self.logging_support and self.keep_log_file:
            os.makedirs(log_dir_path, exist_ok=True)
        if self.batch_docking_support:
            os.makedirs(tmp_ligand_dir_path, exist_ok=True)
            os.makedirs(tmp_output_dir_path, exist_ok=True)
        else:
            os.makedirs(config_dir_path, exist_ok=True)
    
    def __call__(self, smi: Union[str, List[str]], redo_calculation: bool = False) -> List[Union[float, None]]:
        """
        Parameters:
        - smi: SMILES strings to perform docking. A single string activates single-ligand docking mode, while \
            multiple strings utilizes batched docking (if Vina version allows it).
        - redo_calculation: Force redo of ligand preparation, remake config file, and recalculate \
            binding affinity score.
        """
        if type(smi) is str:
            smi = [smi]
        else:
            smi = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True, canonical=True) for s in smi]
        
        if self.batch_docking_support:
            return self._batched_docking(smi)
        else:
            return self._docking(smi)

    def _docking(self, smis: Iterable[str]):
        ligand_path_fn = lambda smi: os.path.abspath(f"{self.ligand_dir_path}{sanitize_smi_name_for_file(smi)}.pdbqt")
        output_path_fn = lambda smi: os.path.abspath(f"{self.output_dir_path}{sanitize_smi_name_for_file(smi)}_out.pdbqt")
        config_path_fn = lambda smi: os.path.abspath(f"{self.config_dir_path}{sanitize_smi_name_for_file(smi)}.conf")
        log_path_fn = lambda smi: os.path.abspath(f"{self.log_dir_path}{sanitize_smi_name_for_file(smi)}_log.txt")

        # NOTE 99% sure this implementation will give unnoticeable parallelism errors if keep_output_file is False so beware
        # NOTE solution: save output to tmp folder and then move to main output folder
        overlap = [i for i in range(len(smis)) if os.path.exists(output_path_fn(smis[i]))]
        ligand_paths = [ligand_path_fn(smis[i]) for i in range(len(smis)) if i not in overlap]
        config_paths = [config_path_fn(smis[i]) for i in range(len(smis)) if i not in overlap]
        if self.logging_support and self.keep_log_file:
            log_paths = [log_path_fn(smis[i]) for i in range(len(smis)) if i not in overlap]
        output_paths = [output_path_fn(smi) for smi in smis]
        output_paths_no_overlap = [output_paths[i] for i in range(len(output_paths)) if i not in overlap]
        
        if self.debug:
            new_ligand_files = self.preparation_profiler.time_it(self._prepare_ligands, smis, ligand_paths)
        else:
            new_ligand_files = self._prepare_ligands(smis, ligand_paths)

        for i in range(len(ligand_paths)):
            self._write_conf_file(config_paths[i], args={"ligand": ligand_paths[i], "out": output_paths_no_overlap[i]})

            if self.logging_support and self.keep_log_file:
                log_path = log_paths[i]
            else:
                log_path = None

            if self.debug:
                self.docking_profiler.time_it(self._run_vina, config_paths[i], log_path)
            else:
                self._run_vina(config_paths[i], log_path)
        
        binding_scores = []
        for output_path in output_paths:
            binding_scores.append(self._get_output_score(output_path))
        
        for i in range(len(ligand_paths)):
            if not self.keep_config_file and os.path.exists(config_paths[i]):
                os.remove(config_paths[i])
            if not self.keep_ligand_file and os.path.exists(ligand_paths[i]) and new_ligand_files[i]:
                os.remove(ligand_paths[i])
            if not self.keep_output_file and os.path.exists(output_paths_no_overlap[i]):
                os.remove(output_paths[i])

        return binding_scores

    def _batched_docking(self, smis: Iterable[str]) -> List[Union[float, None]]:
        # TODO: GPU versions can be made faster by better pipelining (prepare x number of molecules on CPU while docking at same time on GPU)

        tmp_ligand_path_fn = lambda smi: os.path.abspath(f"{self.tmp_ligand_dir_path}{sanitize_smi_name_for_file(smi)}.pdbqt")
        ligand_path_fn = lambda smi: os.path.abspath(f"{self.ligand_dir_path}{sanitize_smi_name_for_file(smi)}.pdbqt")
        output_path_fn = lambda smi: os.path.abspath(f"{self.output_dir_path}{sanitize_smi_name_for_file(smi)}_out.pdbqt")
        tmp_output_path_fn = lambda smi: os.path.abspath(f"{self.tmp_output_dir_path}{sanitize_smi_name_for_file(smi)}_out.pdbqt")
        
        # not the fastest implementation, but safe if multiple experiments running at same time (with different tmp file paths)
        overlap = [i for i in range(len(smis)) if os.path.exists(output_path_fn(smis[i]))]
        ligand_paths = [ligand_path_fn(smis[i]) for i in range(len(smis)) if i not in overlap]
        tmp_ligand_paths = [tmp_ligand_path_fn(smis[i]) for i in range(len(smis)) if i not in overlap]
        if self.keep_output_file:
            output_paths = [output_path_fn(smi) for smi in smis]
        else:
            output_paths = [output_path_fn(smis[i]) if i in overlap else tmp_output_path_fn(smis[i]) for i in range(len(smis))]

        self._prepare_ligands(smis, ligand_paths, tmp_ligand_paths)
        self._write_conf_file(self.tmp_config_file_path, 
                              {"ligand_directory": self.tmp_ligand_dir_path,
                               "output_directory": self.tmp_output_dir_path})

        # Perform docking procedure
        if self.debug:
            self.docking_profiler.time_it(self._run_vina, self.tmp_config_file_path, None)
        else:
            self._run_vina(self.tmp_config_file_path, None)

        # Remove temporary config file
        if os.path.exists(self.tmp_config_file_path):
            os.remove(self.tmp_config_file_path)
        
        # Move files from temporary to proper directory (or delete if redoing calculation)
        if self.keep_ligand_file:
            move_files_from_dir(self.tmp_ligand_dir_path, self.ligand_dir_path)
        else:
            delete_dir_contents(self.tmp_ligand_dir_path)

        if self.keep_output_file:
            move_files_from_dir(self.tmp_output_dir_path, self.output_dir_path)
        else:
            delete_dir_contents(self.tmp_output_dir_path)
        
        # Gather binding scores
        binding_scores = []
        for i in range(len(smis)):
            binding_scores.append(self._get_output_score(output_paths[i]))
        
        return binding_scores
    
    def _prepare_ligands(self, smis: List[str],
                               ligand_paths: List[str],
                               tmp_ligand_paths: List[str] = None) -> List[bool]:
        # Copying from ligand_paths to tmp_ligand_paths (batched docking)
        if tmp_ligand_paths is not None:
            for i in range(len(ligand_paths)):
                if os.path.isfile(ligand_paths[i]):
                    shutil.copy(ligand_paths[i], tmp_ligand_paths[i])
                    if self.print_msgs:
                        print(f'Ligand file: {ligand_paths[i]!r} already exists, copying to {tmp_ligand_paths[i]!r}')

        # Perform ligand preparation and save to proper path (tmp/non-tmp ligand dir)
        if tmp_ligand_paths is not None:
            save_ligand_path = tmp_ligand_paths
        else:
            save_ligand_path = ligand_paths
        
        if self.debug:
            return self.preparation_profiler.time_it(self.ligand_preparation_fn, smis, save_ligand_path)
        else:
            return self.ligand_preparation_fn(smis, save_ligand_path)

    
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
    
    def _write_conf_file(self, config_file_path: str, args: Dict[str, str] = {}):
        conf = f'receptor = {self.receptor_pdbqt_file}\n' + \
               f'center_x = {self.center_pos[0]}\n' + \
               f'center_y = {self.center_pos[1]}\n' + \
               f'center_z = {self.center_pos[2]}\n' + \
               f'size_x = {self.size[0]}\n' + \
               f'size_y = {self.size[1]}\n' + \
               f'size_z = {self.size[2]}\n'

        for k, v in self.additional_vina_args.items():
            conf += f"{str(k)} = {str(v)} \n"
        
        for k, v in args.items():
            conf += f"{str(k)} = {str(v)} \n"
        
        with open(config_file_path, 'w') as f:
            f.write(conf)

        return conf
    
    def _run_vina(self, config_path, log_path):
        try:
            cmd_str = f"{self.vina_cmd} --config {config_path}"
            if self.logging_support and log_path is not None:
                cmd_str += f" --log {log_path}"
            if not self.print_vina_output:
                cmd_str += " > /dev/null 2>&1"

            with subprocess.Popen(cmd_str, shell=True, start_new_session=True, cwd=self.vina_cwd) as proc:
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


def test_preparator():
    preparation_fn = MeekoLigandPreparator(False, timeout_duration, n_workers=multiprocessing.cpu_count())
    smiles = data['smiles'].iloc[:10000].tolist()
    ligand_files = [f"./ligands/{sanitize_smi_name_for_file(x)}.pdbqt" for x in smiles]
    preparation_fn(smiles, ligand_files)

if __name__ == "__main__":
    data = data = MolGen(name = 'ZINC').get_data()
    timeout_duration = None

    preparation_fn = MeekoLigandPreparator(False, timeout_duration, n_workers=multiprocessing.cpu_count())

    delete_all()

    docking_module = VinaDockingModule("/home/nick/Desktop/MolDocking/QuickVina2-GPU/QuickVina2-GPU",
                                       "advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=True,
                                       keep_config_file=True,
                                       keep_log_file=True,
                                       keep_output_file=True,
                                       print_vina_output=True,
                                       print_msgs=True,
                                       debug=True,
                                       ligand_preparation_fn=preparation_fn,
                                       timeout_duration=timeout_duration,
                                       vina_cwd='/home/nick/Desktop/MolDocking/QuickVina2-GPU/')
    print(docking_module(data['smiles'].iloc[:5]))

