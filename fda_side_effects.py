import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import joblib


# Load the SIDER dataset
sider_file_path = r"C:\Users\sthdh\Downloads\merged_output_final1.csv"
sider_df = pd.read_csv(sider_file_path, encoding="ISO-8859-1")

# Load the new side effects dataset (containing side effect names and scores)
side_effect_scores_path = r"C:\Users\sthdh\Downloads\scoring_sider.xlsx"
side_effect_scores_df = pd.read_excel(side_effect_scores_path)

# Convert side effect names to lowercase for matching
side_effect_scores_df["Name"] = side_effect_scores_df["Name"].str.lower()

# Create dictionaries for severity and frequency
side_effect_severity_dict = dict(zip(side_effect_scores_df["Name"], side_effect_scores_df["Rank score"]))
side_effect_freq_dict = dict(zip(side_effect_scores_df["Name"], side_effect_scores_df["Rank Stdev (% out 2929)"]))

# Global dictionary to store FDA-approved drug names and their respective SMILES strings.
fda_drugs = {}  # Keys: drug generic name, Value: SMILES

def get_smiles_from_generic_name(generic_name):
    """Fetches the SMILES representation of a drug from its generic name using the PubChem API."""
    try:
        formatted_name = generic_name.replace(" ", "+")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{formatted_name}/property/CanonicalSMILES/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return smiles
        else:
            return None  
    except Exception:
        return None 

def fetch_drug_info(disease_name, flag):
    base_url = "https://api.fda.gov/drug/label.json"
    params = {"search": f"indications_and_usage:{disease_name}", "limit": 30}
    filtered_drugs = []
    seen_generic_names = set()  # to avoid duplicates
    fda_drugs = {}  # <---- ADDED HERE

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                for result in data["results"]:
                    drug_info = result.get("openfda", {})
                    brand_name = drug_info.get("brand_name", ["Unknown"])[0]
                    generic_name = drug_info.get("generic_name", ["Unknown"])[0]
                    product_ndc = drug_info.get("product_ndc", ["Unknown"])[0]
                    unii = drug_info.get("unii", ["Unknown"])[0]

                    # Skip duplicates (by generic name)
                    if generic_name in seen_generic_names:
                        continue

                    if all(value != "Unknown" for value in [brand_name, generic_name, product_ndc, unii]):
                        smiles = get_smiles_from_generic_name(generic_name)
                        if smiles:  # Only consider drugs with a valid SMILES.
                            fda_drugs[generic_name] = smiles  # <---- STORE HERE

                            # Filter side effects from SIDER (matching on generic name, using PT)
                            side_effects = sider_df.loc[
                                (sider_df["name"].str.lower() == generic_name.lower()) & 
                                (sider_df["pref_left"] == "PT"),
                                "se"
                            ]
                            side_effects_list = side_effects.dropna().tolist()

                            cumulative_score = 0
                            valid_count = 0
                            side_effects_with_scores = []
                            for se in side_effects_list:
                                se_lower = se.lower()
                                severity = side_effect_severity_dict.get(se_lower)
                                frequency = side_effect_freq_dict.get(se_lower)
                                if severity is not None and frequency is not None:
                                    product = severity * frequency
                                    cumulative_score += product
                                    valid_count += 1
                                    side_effects_with_scores.append(
                                        f"{se} (Severity: {severity}, Frequency: {frequency}, Product: {product})"
                                    )
                                else:
                                    side_effects_with_scores.append(f"{se} (Score not available)")

                            if valid_count > 0:
                                normalized_score = cumulative_score / valid_count
                            else:
                                normalized_score = None

                            if side_effects_with_scores:
                                side_effects_text = ", ".join(side_effects_with_scores)
                            else:
                                side_effects_text = "Side effects not found in SIDER"

                            filtered_drugs.append({
                                "Brand Name": brand_name,
                                "Generic Name": generic_name,
                                "Product NDC": product_ndc,
                                "UNII": unii,
                                "SMILES": smiles,
                                "Side Effects": side_effects_text,
                                "Count of side effects": valid_count,
                                "Normalized Cumulative Score": normalized_score
                            })

                            seen_generic_names.add(generic_name)

                if not filtered_drugs:
                    return [f"No FDA-approved drugs with SMILES found for {disease_name}."], {}

                if flag == 0:
                    sorted_drugs = sorted(
                        filtered_drugs,
                        key=lambda x: x["Normalized Cumulative Score"] if x["Normalized Cumulative Score"] is not None else float('inf')
                    )
                    final_ranking = []
                    for idx, drug in enumerate(sorted_drugs, 1):
                        final_ranking.append({
                            "Rank": idx,
                            "Generic Name": drug["Generic Name"],
                            "Normalized Cumulative Score": drug["Normalized Cumulative Score"]
                        })
                    return final_ranking, fda_drugs  # <--- FIXED RETURN
                else:
                    return filtered_drugs, fda_drugs  # <--- FIXED RETURN
            else:
                return [f"No FDA-approved drugs found for {disease_name}."], {}
        else:
            return [f"Error: {response.status_code} - {response.text}"], {}
    except Exception as e:
        return [f"An error occurred: {e}"], {}
