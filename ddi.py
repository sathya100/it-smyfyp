import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, DataStructs
from itertools import product

def get_molecule_features(smiles):
    """
    Computes a set of molecular features for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol_wt = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    
    return {
        'MolWt': mol_wt,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'TPSA': tpsa,
        'Rotatable': rotatable_bonds,
        'Fingerprint': fingerprint
    }

def calculate_similarity(fp1, fp2):
    """Computes Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_all_features(smiles1, smiles2):
    """
    Extracts features for two molecules and returns a dictionary including their fingerprint similarity.
    """
    feat1 = get_molecule_features(smiles1)
    feat2 = get_molecule_features(smiles2)
    similarity = calculate_similarity(feat1['Fingerprint'], feat2['Fingerprint'])
    return {
        'MolWt_1': feat1['MolWt'],
        'MolWt_2': feat2['MolWt'],
        'LogP_1': feat1['LogP'],
        'LogP_2': feat2['LogP'],
        'HBD_1': feat1['HBD'],
        'HBD_2': feat2['HBD'],
        'HBA_1': feat1['HBA'],
        'HBA_2': feat2['HBA'],
        'TPSA_1': feat1['TPSA'],
        'TPSA_2': feat2['TPSA'],
        'RotatableBonds_1': feat1['Rotatable'],
        'RotatableBonds_2': feat2['Rotatable'],
        'Fingerprint_Similarity': similarity
    }

def predict_for_all_combinations(cancer_drugs, fda_drugs):
    # Load the pre-trained model and mapping CSV.
    clf = joblib.load(r"D:\final_conversion\final_conversion\random_forest_model_final_bro.pkl")
    df_mapping = pd.read_csv(r"D:\final_conversion\final_conversion\mapping - aggregated_data.csv")
    
    results = []
    # Loop over all combinations using itertools.product:
    for cancer_smiles, (fda_name, fda_smiles) in product(cancer_drugs, fda_drugs.items()):
        try:
            features = get_all_features(cancer_smiles, fda_smiles)
            X_input = pd.DataFrame([features])
            predicted_y = clf.predict(X_input)[0]
            # Lookup corresponding Map1 values
            matching_rows = df_mapping[df_mapping['Y'] == predicted_y]
            map1_values = matching_rows['Map1'].unique()
            map1_str = ", ".join(map(str, map1_values))
            
            results.append({
                "Cancer_Drug": cancer_smiles,
                "FDA_Drug": fda_name,
                "FDA_SMILES": fda_smiles,
                "Predicted_Y": predicted_y,
                "Map1": map1_str
            })
        except Exception as e:
            results.append({
                "Cancer_Drug": cancer_smiles,
                "FDA_Drug": fda_name,
                "FDA_SMILES": fda_smiles,
                "Predicted_Y": None,
                "Map1": str(e)
            })
    
    return pd.DataFrame(results).sort_values(by="Map1", ascending=True, na_position="last").reset_index(drop=True)