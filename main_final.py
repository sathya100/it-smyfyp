#!/usr/bin/env python3
import sys
print("Using Python interpreter:", sys.executable)
import pandas as pd
# main.py
# Importing functions from your converted modules.
import quantum_modell         
import cancer_admet_module   
import cancer_combined_ranking  
import docking               
import fda_admet            
import fda_side_effects     
import fda_combined_ranking  
import ddi                  
import final_rank           

def main():
    # Get user input for protein sequence and number of drugs.
    protein_input = input("Enter protein sequence: ").strip()
    try:
        n_drugs = int(input("Enter number of drugs to consider: ").strip())
    except ValueError:
        print("Invalid number entered. Exiting.")
        sys.exit()
    
    # Load the KIBA dataset and filter based on protein input.
    kiba_file_path = r"C:\Users\sthdh\Downloads\KIBA.csv\KIBA.csv"  # Update the path as needed.
    kiba_df = pd.read_csv(kiba_file_path)
    kiba_df = kiba_df[kiba_df["target_sequence"].str.contains(protein_input, case=False, na=False)]
    if kiba_df.empty:
        print(f"No drugs found for protein '{protein_input}' in the KIBA dataset.")
        sys.exit()

    # Extract SMILES representations (or drug identifiers) for the drugs.
    drug_smiles_list = kiba_df["compound_iso_smiles"].dropna().unique().tolist()

    print(f"\nComputing top {n_drugs} drugs for protein based on binding affinity using QNN...")
    # Use the quantum model function to get binding affinity predictions.
    predictions = quantum_modell.predict_binding_affinities(drug_smiles_list, protein_input)
    
    # Sort predictions (assuming higher predicted affinity values are better).
    sorted_predictions = sorted(predictions, key=lambda x: x["predicted_affinity"], reverse=True)
    top_n_cancer_predictions = sorted_predictions[:n_drugs]
    
    binding_affinity_dict = {}
    print(f"\nTop {n_drugs} drugs for protein:")
    for rank, entry in enumerate(top_n_cancer_predictions, start=1):
        print(f"Rank {rank}: Drug: {entry['drug']} => Predicted Affinity: {entry['predicted_affinity']:.4f}")
        binding_affinity_dict[entry['drug']] = entry['predicted_affinity']

    # Extract the list of top cancer drug identifiers.
    cancer_drug_smiles_list = [entry["drug"] for entry in top_n_cancer_predictions]

    # Perform dummy docking analysis.
    protein_path=r"C:\Users\sthdh\OneDrive\Desktop\fiidock\P17612.pdbqt"
    dock_final_predictions = docking.perform_docking(cancer_drug_smiles_list, protein_path)
    print("\nRanking of Drugs based on number of docking sites:")
    
    dock_dict = {}
    current_rank = 1
    prev_sites = None
    for entry in dock_final_predictions:
        if prev_sites is not None and entry['active_torsions'] != prev_sites:
            current_rank += 1
        print(f"Rank {current_rank}: Drug: {entry['drug']} => No. of cavities: {entry['active_torsions']:.4f}")
        dock_dict[entry['drug']] = entry['active_torsions']
        prev_sites = entry['active_torsions']
    
    # Obtain cancer ADMET analysis results.
    cancer_admet_dict = cancer_admet_module.cancer_admet_analysis(cancer_drug_smiles_list, flag=0)
    
    # Combine the results using TOPSIS ranking.
    cancer_final_ranking = cancer_combined_ranking.cancer_final_combined_topsis(
                                binding_affinity_dict, cancer_admet_dict, dock_dict)
    
    print("\nFinal Combined TOPSIS Ranking of Cancer Drugs (Higher Combined Score is Better):")
    for item in cancer_final_ranking:
        print(f"Rank {item['Rank']}: Drug: {item['Drug']} => Combined Score: {item['Combined Score']:.4f}")
    
    cancer_ranking_df = pd.DataFrame(cancer_final_ranking)
    print("\nCancer Drug Ranking using TOPSIS:")
    print(cancer_ranking_df)
    
    # ----- FDA Drug Analysis -----
    disease = input("Enter a disease name: ").strip()
    
    print("\nFetching FDA approved drugs...")
    # Get side effect based ranking and fda_drugs dictionary
    side_effect_result, fda_drugs = fda_side_effects.fetch_drug_info(disease, 0)
    
    print(f"\nList of FDA approved drugs for {disease}:")
    if fda_drugs:
        for i, drug in enumerate(fda_drugs.keys(), start=1):
            print(f"{i}. {drug}")
    else:
        print("No FDA drug data available.")
    
    print("\nRanking of Drugs based on severity of side effects:")
    if isinstance(side_effect_result, list):
        for item in side_effect_result:
            print(item)
    else:
        print(side_effect_result)
    
    # Build dictionary: drug name -> normalized cumulative side effect score.
    side_effect_scores = {
        item["Generic Name"]: item["Normalized Cumulative Score"]
        for item in side_effect_result
        if "Normalized Cumulative Score" in item
    }
    
    # Get ADMET analysis scores for FDA drugs.
    fda_admet_scores = fda_admet.fda_admet_analysis(fda_drugs, 0)
    
    # Combine FDA drug results using TOPSIS.
    fda_final_ranking = fda_combined_ranking.fda_final_combined_topsis(
                            side_effect_scores, fda_admet_scores)
    print("\nFinal Combined TOPSIS Ranking for FDA drugs (Side effects & ADMET):")
    for item in fda_final_ranking:
        print(item)
    
    fda_ranking_df = pd.DataFrame(fda_final_ranking)
    print("\nDisease's Drug Ranking using TOPSIS:")
    print(fda_ranking_df)
    
    # ----- Drug-Drug Interaction (DDI) Prediction -----
    ddi_result = ddi.predict_for_all_combinations(cancer_drug_smiles_list, fda_drugs)
    print("\nDrug-Drug Interaction Predictions:")
    print(ddi_result[['Cancer_Drug', 'FDA_Drug', 'Map1']])
    
    # ----- Final Pair TOPSIS Ranking -----
    final_pair_rank = final_rank.final_pair_topsis(cancer_final_ranking, fda_final_ranking, ddi_result)
    final_pair_df = pd.DataFrame(final_pair_rank)
    print("\nFinal Pair TOPSIS Ranking:")
    print(final_pair_df)

if __name__ == '__main__':
    main()
