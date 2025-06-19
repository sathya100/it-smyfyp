import os
import shutil
import stat
import subprocess
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolfiles

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def perform_docking(drug_smiles_list, protein_pdbqt_path):
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    base_dir = os.path.abspath("docking_workflow_temp")
    drug_pdb_dir = os.path.join(base_dir, "drugpdb")
    drug_pdbqt_dir = os.path.join(base_dir, "drugpdbqt")
    outputs_dir = os.path.join(base_dir, "docking_outputs")

    for d in [drug_pdb_dir, drug_pdbqt_dir, outputs_dir]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d, onerror=remove_readonly)
            except Exception:
                pass
        os.makedirs(d, exist_ok=True)

    # Generate PDB files from SMILES
    for idx, smi in enumerate(drug_smiles_list, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        pdb_path = os.path.join(drug_pdb_dir, f"drug_{idx}.pdb")
        with open(pdb_path, 'w') as f:
            f.write(rdmolfiles.MolToPDBBlock(mol))

    pdb_files = [f for f in os.listdir(drug_pdb_dir) if f.lower().endswith(".pdb")]

    try:
        subprocess.run(["obabel", "-V"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return []

    # Convert PDB to PDBQT using Open Babel
    for pdb_file in pdb_files:
        input_path = os.path.join(drug_pdb_dir, pdb_file)
        output_path = os.path.join(drug_pdbqt_dir, os.path.splitext(pdb_file)[0] + ".pdbqt")
        command = ["obabel", input_path, "-O", output_path, "--gen3d"]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue

    # Extract docking box parameters from protein
    xs, ys, zs = [], [], []
    try:
        with open(protein_pdbqt_path, "r") as pf:
            for line in pf:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return []

    if not xs or not ys or not zs:
        return []

    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    center_z = (max(zs) + min(zs)) / 2
    size_x = max(xs) - min(xs) + 10
    size_y = max(ys) - min(ys) + 10
    size_z = max(zs) - min(zs) + 10

    # Create config file for Vina
    config_path = os.path.join(outputs_dir, "config.txt")
    with open(config_path, "w") as cfg:
        cfg.write(f"center_x = {center_x:.2f}\n")
        cfg.write(f"center_y = {center_y:.2f}\n")
        cfg.write(f"center_z = {center_z:.2f}\n\n")
        cfg.write(f"size_x = {size_x:.2f}\n")
        cfg.write(f"size_y = {size_y:.2f}\n")
        cfg.write(f"size_z = {size_z:.2f}\n\n")
        cfg.write("exhaustiveness = 8\n")
        cfg.write("num_modes = 9\n")
        cfg.write("energy_range = 3\n")

    vina_path = r"C:\Program Files (x86)\The Scripps Research Institute\Vina\vina.exe"
    docking_results = []
    ligand_files = [f for f in os.listdir(drug_pdbqt_dir) if f.endswith(".pdbqt")]

    if not ligand_files:
        return []

    for ligand_file in ligand_files:
        ligand_path = os.path.join(drug_pdbqt_dir, ligand_file)
        ligand_name = os.path.splitext(ligand_file)[0]
        log_file = os.path.join(outputs_dir, f"{ligand_name}_log.txt")
        output_file = os.path.join(outputs_dir, f"{ligand_name}_output.pdbqt")

        command = [
            vina_path,
            "--receptor", protein_pdbqt_path,
            "--ligand", ligand_path,
            "--config", config_path,
            "--log", log_file,
            "--out", output_file
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                continue
        except Exception:
            continue

        active_torsions = 0
        try:
            with open(output_file, "r") as outf:
                for line in outf:
                    if "active torsions" in line:
                        tokens = line.strip().split()
                        try:
                            active_torsions = int(tokens[1])
                        except (IndexError, ValueError):
                            active_torsions = 0
                        break
        except FileNotFoundError:
            active_torsions = 0

        try:
            index = int(ligand_name.split("_")[-1]) - 1
            drug_smiles = drug_smiles_list[index]
        except Exception:
            drug_smiles = "Unknown"

        docking_results.append({
            "drug": drug_smiles,
            "active_torsions": active_torsions
        })

    sorted_results = sorted(docking_results, key=lambda x: x["active_torsions"])
    return sorted_results


