import os
import subprocess
import re

import time
from typing import Dict, List, Union
from tdc.generation import MolGen
from tqdm import tqdm


def obabel_ligand_preparation(smi: str, ligand_path: str, timeout_duration: int):
    with subprocess.Popen('obabel -:"' + smi + '" -O "' + ligand_path + '" -h --gen3d > /dev/null 2>&1',
        shell=True, start_new_session=True) as proc:
        try:
            proc.wait(timeout=timeout_duration)
        except subprocess.TimeoutExpired:
            proc.kill()


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
    def __init__(self, vina_cmd: str,
                 receptor_pdbqt_file: str,
                 center_pos: List[float],
                 size: List[float],
                 ligand_dir: str = './ligands/',
                 output_dir: str = './outputs/',
                 log_dir: str = './logs/',
                 config_dir: str = './configs/',
                 keep_ligand_file: bool = True,
                 keep_output_file: bool = True,
                 keep_log_file: bool = True,
                 keep_config_file: bool = True,
                 timeout_duration: int = 600,
                 additional_vina_args: Dict[str, str] = None,
                 ligand_preparation_fn: callable = obabel_ligand_preparation,
                 print_msgs: bool = False,
                 debug: bool = False) -> None:
        """
            Parameters:
            - vina_cmd: Command line prefix to execute vina command (e.g. ./qvina2.1)
            - receptor_pdbqt_file: Cleaned receptor PDBQT file to use for docking
            - center_pos: 3-dim list containing (x,y,z) coordinates of grid box
            - size: 3-dim list containing sizing information of grid box in (x,y,z) directions
            - ligand_dir: Path to save ligand preparation files
            - output_dir: Path to save docking output files
            - log_dir: Path to save log files
            - config_dir: Path to save config files
            - keep_ligand_file: Save ligand file (True) or not (False)
            - keep_output_file: Save output file (True) or not (False)
            - keep_log_file: Save log file (True) or not (False)
            - keep_config_file: Save config file (True) or not (False)
            - timeout_duration: Timeout in seconds before new process automatically stops
            - additional_vina_args: Dictionary of additional Vina arguments (e.g. {"cpu": "5"})
            - ligand_preparation_fn: Function to prepare molecule for docking. Should take the \
                argument format (smiles string, ligand_path, timeout_duration)
            - print_msgs: Print messages to console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
        """
        if not os.path.isfile(receptor_pdbqt_file):
            raise Exception(rf'Protein file: {receptor_pdbqt_file} not found')
        
        if len(center_pos) != 3:
            raise Exception(f"center_pos must contain 3 values, {center_pos} was provided")

        if len(size) != 3:
            raise Exception(f"size must contain 3 values, {size} was provided")

        self.vina_cmd = vina_cmd
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center_pos = center_pos
        self.size = size
        self.ligand_dir = ligand_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.config_dir = config_dir
        self.keep_ligand_file = keep_ligand_file
        self.keep_output_file = keep_output_file
        self.keep_log_file = keep_log_file
        self.keep_config_file = keep_config_file
        self.timeout_duration = timeout_duration
        self.additional_vina_args = additional_vina_args
        self.ligand_preparation_fn = ligand_preparation_fn  # should follow format (smiles string, ligand_path, timeout_duration)
        # TODO: lig_prepare_fn needs standardizing
        self.print_msgs = print_msgs  # TODO: include logging levels (print statements, console statements, etc.)
        self.debug = debug

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ligand_dir, exist_ok=True)
    
    def __call__(self, smi: str, redo_calculation: bool = False) -> Union[int, None]:
        ligand_name = smi.replace('(', '{').replace(')', '}').replace('\\', r'%5C').replace('/', r'%2F').replace('#', '%23')  # SMILES string corresponds with file name
        ligand_path = self.ligand_dir + ligand_name + '.pdbqt'
        output_path = self.output_dir + ligand_name + '_out.pdbqt'
        config_path = self.config_dir + ligand_name + '.conf'
        log_path = self.log_dir + ligand_name + '_log.txt'

        # Prepare ligand for docking
        if not os.path.isfile(ligand_path) or redo_calculation:
            if self.debug:
                self.preparation_profiler.time_it(self.ligand_preparation_fn, smi, ligand_path, self.timeout_duration)
            else:
                self.ligand_preparation_fn(smi, ligand_path, self.timeout_duration)
        elif self.print_msgs:
            print(f'Ligand file: {ligand_path!r} already exists')

        if not os.path.isfile(output_path):
            conf = 'receptor = ' + self.receptor_pdbqt_file + '\n' + \
                   'ligand = ' + str(ligand_path) + '\n' + \
                   'center_x = ' + str(self.center_pos[0]) + '\n' + \
                   'center_y = ' + str(self.center_pos[1]) + '\n' + \
                   'center_z = ' + str(self.center_pos[2]) + '\n' + \
                   'size_x = ' + str(self.size[0]) + '\n' + \
                   'size_y = ' + str(self.size[1]) + '\n' + \
                   'size_z = ' + str(self.size[2]) + '\n' + \
                   'out = ' + output_path + '\n'
            
            if self.additional_vina_args is not None:
                for k, v in self.additional_vina_args.items():
                    conf += f"{str(k)} = {str(v)} \n"

            with open(config_path, 'w') as f:
                f.write(conf)
        
        if self.debug:
            self.docking_profiler.time_it(self._run_vina, config_path, log_path)
        else:
            self._run_vina(config_path, log_path)
        
        if not self.keep_ligand_file:
            os.remove(ligand_path)
        if not self.keep_log_file:
            os.remove(log_path)
        if not self.keep_config_file:
            os.remove(config_path)
        
        try:
            score = float("inf")
            with open(output_path, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
            
            if not self.keep_output_file:
                os.remove(output_path)
            
            return score
        except FileNotFoundError:
            return None
    
    def _run_vina(self, config_path, log_path):
        try:
            with subprocess.Popen(self.vina_cmd + \
                    ' --config ' + config_path + \
                    ' --log ' + log_path + \
                    '> /dev/null 2>&1', #> /dev/null 2>&1,
                    stderr=subprocess.PIPE,
                    shell=True, start_new_session=True) as proc:
                proc.wait(timeout=self.timeout_duration)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    docking_module = VinaDockingModule("./qvina2.1",
                                       "advina_module/protein_files/6rqu.pdbqt",
                                       [7.750, -14.556, 6.747],
                                       [20, 20, 20],
                                       keep_ligand_file=False,
                                       keep_config_file=False,
                                       keep_log_file=False,
                                       keep_output_file=False,
                                       debug=True)
    data = MolGen(name = 'ZINC').get_data()

    with tqdm(total=1000) as pbar:
        for smi in data.iloc[:1000]['smiles']:
            docking_module(smi)
            pbar.update(1)
            pbar.set_description(f"Lig. Prepare: {docking_module.preparation_profiler.get_average():.2f}, Docking: {docking_module.docking_profiler.get_average():.2f}")
