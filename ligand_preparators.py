from functools import partial
import multiprocessing
import os
import subprocess
from typing import List, Union
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from meeko import MoleculePreparation, PDBQTWriterLegacy

def execute_shell_process(cmd_str: str, error_msg: str = None, print_output: bool = False, timeout_duration: int = None) -> bool:
    with subprocess.Popen(cmd_str, shell=True, start_new_session=False) as proc:
        try:
            if proc.wait(timeout=timeout_duration) != 0:
                if print_output and error_msg is not None:
                    print(error_msg)
                return False
        except subprocess.TimeoutExpired:
            proc.kill()
            return False
    
    return True


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

    def __del__(self):
        self.close()
        self.join()

#TODO: GYPSUM Ligand Preparator https://github.com/durrantlab/gypsum_dl
#TODO: unidock Ligand Preparator https://github.com/dptech-corp/Uni-Dock/tree/main/unidock_tools


class OBabelLigandPreparator(PooledWorkerExecutor):
    """
        Using obabel command line to prepare ligand.
        Installation instructions at https://openbabel.org/docs/Installation/install.html

        NOTE: UNTESTED
    """
    def __init__(self, print_output: bool, timeout_duration: int = None, n_workers: Union[int, None] = -1) -> None:
        # Checking if obabel available
        available = False
        with subprocess.Popen(f"obabel -V",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True) as proc:
            if proc.wait(timeout=timeout_duration) == 0:
                result = proc.stdout.read()
                if result is not None and "Open Babel" in str(result):
                    available = True
        
        if not available:
            raise Exception("obabel is not installed")
    
        self.print_output = print_output
        self.timeout_duration = timeout_duration
        
        if n_workers != -1:
            self.pool_executor = PooledWorkerExecutor(n_workers, timeout_duration)
        else:
            self.pool_executor = None
    
    def __call__(self, smi: Union[str, List[str]], ligand_path: Union[str, List[str]]) -> List[bool]:
        """
            returns: List containing whether new ligand file was created (True) or is already existing (False)
        """
        print(smi, ligand_path)
        if smi is type(str):
            smi = [smi]
        if ligand_path is type(str):
            ligand_path = [ligand_path]
        
        if self.pool_executor is not None:
            tmp = partial(self._prepare_ligand, print_output=self.print_output, timeout_duration=self.timeout_duration)
            res = self.pool_executor.starmap(tmp, zip(smi, ligand_path))
            #res = res.get()
            #print(res)
            return res.get()
        else:
            res = []
            for i in range(len(smi)):
                res.append(self._prepare_ligand(smi[i], ligand_path[i], self.print_output, self.timeout_duration))
            return res
    
    @staticmethod
    def _prepare_ligand(smi: str, ligand_path: str, print_output: bool = False, timeout_duration: int = None) -> bool:
        """
            Note: Potentially unsafe for concurrent application with batched docking.
            Ensure unique ligands for preparation.

            returns: True if new ligand file was produced, False if ligand file already exists
        """
        if not os.path.exists(ligand_path):
            try:
                cmd_str = f'obabel -:"{smi}" -O "{ligand_path}" -h --gen3d'
                if not print_output:
                    cmd_str += ' > /dev/null 2>&1'
                
                return execute_shell_process(cmd_str, f"Error preparing ligand: {smi}", print_output, timeout_duration)
                
            except Exception as e:
                if print_output:
                    print(f"{smi} could not be prepared: {e}")
        return False


class AutoDockLigandPreparator:
    """
        Using AutoDock to prepare ligand (a python3 translated version).
        GitHub repository at https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
        Installation: "pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"
    """
    def __call__(self, smi: str, ligand_path: str) -> None:
        cmd_str = f'prepare_ligand4'
        self._execute_shell_process(cmd_str, f"Error preparing ligand: {smi}")


class MeekoLigandPreparator:
    """
        Using Meeko to prepare ligand.
        GitHub repository: https://github.com/forlilab/Meeko
        Installation: "pip install meeko"
    """
    def __init__(self, print_output: bool, timeout_duration: int = None, n_workers: Union[int, None] = -1) -> None:
        try:
            import meeko
        except ImportError:
            raise Exception("Meeko package isn't installed.")
        
        self.print_output = print_output
        self.timeout_duration = timeout_duration

        if not print_output:
            RDLogger.DisableLog('rdApp.*')
        
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
            try:
                mol = Chem.MolFromSmiles(smi)

                if mol is not None:
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
                else:
                    return False
            except Exception as e:
                if print_output:
                    print(f"{smi} could not be prepared: {e}")
        return False


class RGFNLigandPreparator:
    """
        Using Meeko to prepare ligand.
        GitHub repository: https://github.com/forlilab/Meeko
        Installation: "pip install meeko"
    """
    def __init__(self, print_output: bool, timeout_duration: int = None, n_workers: Union[int, None] = -1) -> None:
        try:
            import meeko
        except ImportError:
            raise Exception("Meeko package isn't installed.")
        
        self.print_output = print_output
        self.timeout_duration = timeout_duration

        if not print_output:
            RDLogger.DisableLog('rdApp.*')
        
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
    def _prepare_ligand(smi: str, ligand_path: str, print_output: bool = False, conformer_attempts: int = 20) -> bool:
        """
            Note: Potentially unsafe for concurrent application with batched docking.
            Ensure unique ligands for preparation.

            returns: True if new ligand file was produced, False if ligand file already exists
        """
        attempt = 0
        pdbqt_string = None

        if not os.path.exists(ligand_path): 
            while pdbqt_string is None and (attempt == 0 or attempt < conformer_attempts):
                attempt += 1

                try:
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(mol)
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(mol)
                    setup = mol_setups[0]
                    pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)

                except Exception as e:
                    print(f"Failed embedding attempt #{attempt} with error: '{e}'.")

            if pdbqt_string is None:
                return False

            with open(ligand_path, "w") as file:
                file.write(pdbqt_string)
            
            return True
        return False
