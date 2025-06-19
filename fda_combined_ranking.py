import numpy as np

def fda_final_combined_topsis(side_effect_dict, admet_dict):

    # Ensure the two dictionaries use the same set of drug names
    common_drugs = [drug for drug in side_effect_dict if drug in admet_dict]
    if not common_drugs:
        print("No common drugs found for final ranking.")
        return []
    
    # Create decision matrix rows: [side_effect_score, admet_score]
    decision_matrix = []
    for drug in common_drugs:
        decision_matrix.append([side_effect_dict[drug], admet_dict[drug]])
    decision_matrix = np.array(decision_matrix)
    
    # Set weights for the two criteria equally
    weights = np.array([0.5, 0.5])
    # Define criteria beneficial: for side effects, lower is better (False); for ADMET, higher is better (True)
    criteria_beneficial = [False, True]
    
    # Normalize each column (vector normalization)
    norm_matrix = decision_matrix / np.linalg.norm(decision_matrix, axis=0)
    weighted_matrix = norm_matrix * weights
    
    ideal_best = np.where(criteria_beneficial, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(criteria_beneficial, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    
    dist_to_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_to_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
    
    relative_closeness = dist_to_worst / (dist_to_best + dist_to_worst)
    
    # Sort drugs based on final relative closeness (higher is better)
    ranking_indices = np.argsort(relative_closeness)[::-1]
    final_ranking = []
    for idx, i in enumerate(ranking_indices, 1):
        final_ranking.append({
            "Rank": idx,
            "Drug Name": common_drugs[i],
            "Combined Score": relative_closeness[i]
        })
    return final_ranking
