import numpy as np
import pandas as pd

def final_pair_topsis(cancer_final_ranking, fda_final_ranking, ddi_df):
   
    # Create a dictionary from cancer_final_ranking mapping cancer drug name --> Combined Score
    cancer_scores = { entry["Drug"]: entry["Combined Score"] for entry in cancer_final_ranking }
    
    # Create a dictionary from fda_final_ranking mapping FDA drug name --> Combined Score
    fda_scores = { entry["Drug Name"]: entry["Combined Score"] for entry in fda_final_ranking }
    
    # Create a list of valid pairs from ddi_df (only those pairs where both drugs appear in our ranking dictionaries)
    valid_pairs = []
    for idx, row in ddi_df.iterrows():
        cancer = row["Cancer_Drug"]
        fda = row["FDA_Drug"]
        # Only include the pair if both exist
        if cancer in cancer_scores and fda in fda_scores:
            valid_pairs.append({
                "Cancer_Drug": cancer,
                "FDA_Drug": fda,
                "Interaction": row["Map1"]
            })
    
    if not valid_pairs:
        print("No valid drug pairs found for combined ranking.")
        return []
    
    # Build the decision matrix:
    # Column 1: Interaction (from ddi_df; non-beneficial: lower is better)
    # Column 2: Cancer Ranking Score (beneficial)
    # Column 3: FDA Ranking Score (beneficial)
    decision_matrix = []
    pair_ids = []  # Store pair identifiers for later use.
    for pair in valid_pairs:
        cancer = pair["Cancer_Drug"]
        fda = pair["FDA_Drug"]
        # Explicit conversion to float for each criterion.
        interaction = float(pair["Interaction"])
        cancer_score = float(cancer_scores[cancer])
        fda_score = float(fda_scores[fda])
        decision_matrix.append([interaction, cancer_score, fda_score])
        pair_ids.append((cancer, fda))
    
    decision_matrix = np.array(decision_matrix)
    
    # Beneficial flags: for Interaction, lower is better (False); for cancer and FDA scores, higher is better (True).
    beneficial = [False, True, True]
    
    # Use equal weights for all three criteria (adjust as needed)
    weights = np.array([1/3, 1/3, 1/3])
    
    # Normalize each column (vector normalization)
    norm_matrix = decision_matrix / np.linalg.norm(decision_matrix, axis=0)
    
    # Multiply by weights
    weighted_matrix = norm_matrix * weights
    
    # Determine ideal best and worst for each criterion.
    # For beneficial criteria: ideal best = max, ideal worst = min.
    # For non-beneficial (interaction): ideal best = min, ideal worst = max.
    ideal_best = np.where(np.array(beneficial), weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(np.array(beneficial), weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    
    # Calculate Euclidean distances to ideal best and worst.
    dist_to_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_to_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
    
    # Calculate relative closeness to ideal solution.
    overall_scores = dist_to_worst / (dist_to_best + dist_to_worst)
    
    # Sort pairs by overall score (higher is better).
    sorted_indices = np.argsort(overall_scores)[::-1]
    
    final_ranking = []
    for rank, idx in enumerate(sorted_indices, start=1):
        cancer, fda = pair_ids[idx]
        final_ranking.append({
            "Rank": rank,
            "Cancer_Drug": cancer,
            "FDA_Drug": fda,
            "Overall Score": overall_scores[idx]
        })
    
    return final_ranking

