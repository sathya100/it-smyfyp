# admet_module.py
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load the scaler and trained ADMET model
scaler = joblib.load(r'D:\final_conversion\final_conversion\admet_scaler.pkl')
best_model = joblib.load(r'D:\final_conversion\final_conversion\admet_model.pkl')

# Define ADMET properties and criteria.
full_properties = [
    'HIA_Hou', 'PAMPA_NCATS', 'Pgp_Broccatelli', 'Caco2_Wang', 'BBB_Martins', 'Bioavailability_Ma',
    'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Substrate_CarbonMangels',
    'CYP2C9_Veith', 'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Substrate_CarbonMangels',
    'CYP3A4_Veith', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach', 'VDss_Lombardo',
    'AMES', 'Carcinogens_Lagunin', 'ClinTox', 'DILI', 'hERG', 'Skin_Reaction', 'molecular_weight',
    'logP', 'hydrogen_bond_acceptors', 'hydrogen_bond_donors', 'Lipinski', 'QED', 'stereo_centers', 'tpsa'
]

criteria_beneficial = [
    True, True, False, True, True, True, False, True, False, False, False, False, False, False, False, False,
    True, True, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False
]

# Indices for higher importance and custom weights.
higher_importance = {0, 3, 5, 8, 9, 11, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 31}
custom_weights = np.array([1.0 if i in higher_importance else 0.5 for i in range(len(full_properties))])
custom_weights = custom_weights / np.sum(custom_weights)

def extract_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        mol.GetNumAtoms(), Descriptors.HeavyAtomCount(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumHeteroatoms(mol), Descriptors.MolMR(mol), Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol), Descriptors.Kappa3(mol), Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol)
    ]

def cancer_admet_analysis(drug_input, flag):
    """
    Input: drug_input is a list of SMILES strings.
    If flag == 0, performs TOPSIS ranking and returns a dictionary mapping SMILES to ADMET relative closeness.
    If flag == 1, prints full ADMET details and returns an empty dict.
    """
    feature_list = []
    valid_smiles = []
    for s in drug_input:
        desc = extract_descriptors(s)
        if desc is not None:
            feature_list.append(desc)
            valid_smiles.append(s)
    if len(feature_list) == 0:
        print("No valid molecular descriptors extracted.")
        return {}
    new_features = pd.DataFrame(feature_list, columns=[
        'MolWt', 'LogP', 'TPSA', 'HDonorCount', 'HAcceptorCount', 'NumRotatableBonds',
        'NumAtoms', 'HeavyAtomCount', 'NumAromaticRings', 'NumHeteroatoms', 'MolMR',
        'Kappa1', 'Kappa2', 'Kappa3', 'Fraction_Csp3', 'NumRings'
    ])
    new_features_scaled = scaler.transform(new_features)
    predictions = best_model.predict(new_features_scaled)
    
    if flag == 1:
        print("ADMET Properties for Each Drug:")
        for idx, s in enumerate(valid_smiles):
            print(f"\nDrug {idx+1}: {s}")
            for prop, val in zip(full_properties, predictions[idx]):
                print(f"  {prop}: {val:.4f}")
        return {}
    
    decision_matrix = np.array(predictions)
    norm_matrix = decision_matrix / np.linalg.norm(decision_matrix, axis=0)
    weighted_matrix = norm_matrix * custom_weights
    ideal_best = np.where(np.array(criteria_beneficial), weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(np.array(criteria_beneficial), weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    dist_to_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_to_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
    relative_closeness = dist_to_worst / (dist_to_best + dist_to_worst)
    admet_scores = { valid_smiles[i]: relative_closeness[i] for i in range(len(valid_smiles)) }
    
    print("\nTOPSIS Ranking of Drugs based on ADMET properties:")
    ranking_indices = np.argsort(relative_closeness)[::-1]
    for rank, i in enumerate(ranking_indices, 1):
        print(f"Rank {rank}: SMILES: {valid_smiles[i]} - Relative Closeness: {relative_closeness[i]:.4f}")
    
    return admet_scores

