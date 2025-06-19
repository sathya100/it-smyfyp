import numpy as np

def cancer_final_combined_topsis(binding_affinity_dict, admet_dict, dock_dict):
    # Get common drugs present in all three dictionaries
    common_drugs = [drug for drug in binding_affinity_dict if drug in admet_dict and drug in dock_dict]
    
    if not common_drugs:
        print("No common drugs found for final combined ranking.")
        return []
    
    # Construct decision matrix [affinity, admet, docking]
    decision_matrix = []
    for drug in common_drugs:
        decision_matrix.append([
            binding_affinity_dict[drug], 
            admet_dict[drug], 
            dock_dict[drug]
        ])
    
    decision_matrix = np.array(decision_matrix)
    
    # All criteria are beneficial: higher is better
    criteria_beneficial = [True, True, True]
    
    # Equal weights for all three criteria
    weights = np.array([1/3, 1/3, 1/3])
    
    # Normalize decision matrix
    norm_matrix = decision_matrix / np.linalg.norm(decision_matrix, axis=0)
    
    # Weight the normalized matrix
    weighted_matrix = norm_matrix * weights
    
    # Determine ideal best and worst for each criterion
    ideal_best = np.where(criteria_beneficial, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(criteria_beneficial, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    
    # Calculate distance to ideal best and worst
    dist_to_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_to_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
    
    # Calculate relative closeness to ideal solution
    relative_closeness = dist_to_worst / (dist_to_best + dist_to_worst)
    
    # Rank based on relative closeness (higher is better)
    ranking_indices = np.argsort(relative_closeness)[::-1]
    
    final_ranking = []
    for rank, idx in enumerate(ranking_indices, 1):
        final_ranking.append({
            "Rank": rank,
            "Drug": common_drugs[idx],
            "Combined Score": relative_closeness[idx]
        })
    
    return final_ranking
