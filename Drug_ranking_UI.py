'''import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Imported modules from your project
import quantum_model
import cancer_admet_module
import cancer_combined_ranking
import docking
import fda_admet
import fda_side_effects
import fda_combined_ranking
import ddi
import final_rank

# Helper to move Rank column to the front
def reorder_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "Rank" in df.columns:
        return df[["Rank"] + [c for c in df.columns if c != "Rank"]]
    return df

# Configure page layout
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Button styles */
    .page-link-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        border: 2px solid black;  /* Black border */
        border-radius: 12px;
        background-color: #939c92;
        color: black;  /* Black text */
        font-weight: 600;
        text-align: center;
        text-decoration: none;
        transition: 0.3s ease;
    }

    .page-link-button:hover {
        background-color: #939c92;
        color: black;  /* Keep text black on hover */
        border-color: black;  /* Keep border black on hover */
    }

    .stButton>button {
        height: auto;
    }

    /* Global padding adjustments */
    div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 ")
st.write(" ")
st.write(" ")



# Top nav buttons
cols = st.columns(5)
pages = [
    ("pages/Quantum.py","Quantum"),
    ("pages/ADMET.py","ADMET"),
    ("pages/SE.py","Side Effect Severity"),
    ("pages/Docking_ui.py","Active Torsion Sites"),
    ("pages/DDI_ui.py","Drug‑Drug Interaction"),
]

for col, (path, label) in zip(cols, pages):
    with col:
        st.markdown(f'<a href="{path}" class="page-link-button">{label}</a>', unsafe_allow_html=True)

st.write("---")


st.title("Drug Ranking System for Co-Morbid Patients")

# Input form for both Cancer and FDA analysis
with st.form(key="analysis_form"):
    protein_input = st.text_input("Enter protein sequence:")
    n_drugs = st.number_input("Number of drugs to consider:", min_value=1, step=1)
    disease = st.text_input("Enter a disease name :")
    submit_button = st.form_submit_button(label="Run Analysis")

if submit_button:
     with st.spinner("Analyzing...."):
    # --- Cancer Analysis ---

        kiba_file_path = "/Users/diyaarshiya/Desktop/KIBA.csv"
        kiba_df = pd.read_csv(kiba_file_path)
        kiba_df = kiba_df[kiba_df["target_sequence"].str.contains(protein_input, case=False, na=False)]
        if kiba_df.empty:
            st.error(f"No drugs found for protein '{protein_input}'.")
        else:
            # Quantum predictions & ranking
            drug_smiles_list = kiba_df["compound_iso_smiles"].dropna().unique().tolist()
            predictions = quantum_model.predict_binding_affinities(drug_smiles_list, protein_input)
            sorted_predictions = sorted(predictions, key=lambda x: x["predicted_affinity"], reverse=True)
            top_n_cancer = sorted_predictions[:n_drugs]
            for idx, item in enumerate(top_n_cancer):
                item['Rank'] = idx + 1
            binding_affinity_dict = {e['drug']: e['predicted_affinity'] for e in top_n_cancer}

            # Docking with dense ranking for ties
            cancer_drug_smiles_list = [e['drug'] for e in top_n_cancer]
            dock_results = docking.dummy_docking(cancer_drug_smiles_list, protein_input)
            dock_results = sorted(dock_results, key=lambda x: x['sites'], reverse=True)
            prev_sites, rank = None, 0
            for item in dock_results:
                if item['sites'] != prev_sites:
                    rank += 1
                    prev_sites = item['sites']
                item['Rank'] = rank
            dock_dict = {e['drug']: e['sites'] for e in dock_results}

            # Cancer ADMET and ranking
            cancer_admet_dict = cancer_admet_module.cancer_admet_analysis(cancer_drug_smiles_list, flag=0)
            cancer_admet_df = pd.DataFrame(cancer_admet_dict.items(), columns=["Drug", "ADMET Score"])
            cancer_admet_df = cancer_admet_df.sort_values(by="ADMET Score", ascending=False).reset_index(drop=True)
            cancer_admet_df.insert(0, "Rank", [i+1 for i in range(len(cancer_admet_df))])

            # Final combined TOPSIS ranking
            cancer_final = cancer_combined_ranking.cancer_final_combined_topsis(
                binding_affinity_dict, cancer_admet_dict, dock_dict
            )
            for idx, item in enumerate(cancer_final):
                item['Rank'] = idx + 1
            drugs = [item['Drug'] for item in cancer_final]

            # --- FDA Analysis ---
            side_effects, fda_drugs = fda_side_effects.fetch_drug_info(disease, 0)
            side_effect_scores = {
                i['Generic Name']: i['Normalized Cumulative Score']
                for i in side_effects if 'Normalized Cumulative Score' in i
            }
            fda_admet_scores = fda_admet.fda_admet_analysis(fda_drugs, 0)
            fda_final = fda_combined_ranking.fda_final_combined_topsis(
                side_effect_scores, fda_admet_scores
            )
            for idx, item in enumerate(fda_final):
                item['Rank'] = idx + 1
            fda_drug_list = list(fda_drugs.keys())

            # Build FDA ADMET DataFrame
            fda_admet_df = pd.DataFrame(fda_admet_scores.items(), columns=["Drug", "ADMET Score"])
            fda_admet_df.insert(0, "Rank", [i+1 for i in range(len(fda_admet_df))])

            # --- Layout: three columns with a thin divider ---
            col1, col_div, col2 = st.columns([5, 0.5, 5])

            with col1:
                
                st.success("CANCER DRUG RANKINGS")
                # Top‑N candidates
                st.subheader("Ranking based on Binding Affinity")
                df_top = pd.DataFrame(top_n_cancer)
                st.dataframe(reorder_rank(df_top), height=200)

                # Docking results
                st.subheader("Ranking based on no. of Active Torsions")
                df_dock = pd.DataFrame(dock_results)
                st.dataframe(reorder_rank(df_dock), height=200)

                # ADMET results
                st.subheader("Ranking based on ADMET Scores")
                st.dataframe(reorder_rank(cancer_admet_df), height=200)

                # Final combined ranking
                st.subheader("Intermediate Cancer Drug Ranking")
                df_final_cancer = pd.DataFrame(cancer_final)
                st.dataframe(reorder_rank(df_final_cancer), height=200)

                # Cancer ranking plot
                fig1, ax1 = plt.subplots(figsize=(6, 7))
                x = np.arange(4)
                crit = ['Affinity', 'Docking', 'ADMET', 'Combined']
                # compute individual criterion ranks
                rank_c1 = {e['drug']: idx+1 for idx, e in enumerate(sorted_predictions)}
                curr, prev = 1, None
                rank_c2 = {}
                for e in dock_results:
                    if prev is not None and e['sites'] != prev:
                        curr += 1
                    rank_c2[e['drug']] = curr
                    prev = e['sites']
                sorted_admet = sorted(cancer_admet_dict.items(), key=lambda x: x[1], reverse=True)
                rank_c3 = {d: idx+1 for idx, (d, _) in enumerate(sorted_admet)}
                rank_c4 = {item['Drug']: item['Rank'] for item in cancer_final}
                for d in drugs:
                    ax1.plot(x, [rank_c1[d], rank_c2[d], rank_c3[d], rank_c4[d]],
                            marker='o', label=d)
                ax1.set_xticks(x)
                ax1.set_xticklabels(crit)
                ax1.set_ylabel('Rank')
                ax1.set_title('Cancer Drug Ranks by Criteria')
                ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), frameon=False)
                st.pyplot(fig1)

            with col_div:
                st.markdown(
                    "<div style='border-left:2px solid black; height:100%;'></div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.success("FDA DRUG RANKINGS")

                # Side effects (no Rank to reorder)
                st.subheader("Ranking based on Side Effects Severity")
                st.dataframe(pd.DataFrame(side_effects), height=200)

                # FDA ADMET
                st.subheader("Ranking based on ADMET Scores")
                st.dataframe(reorder_rank(fda_admet_df), height=200)

                # Final combined FDA ranking
                st.subheader("Intermediate Disease Drug Ranking")
                df_final_fda = pd.DataFrame(fda_final)
                st.dataframe(reorder_rank(df_final_fda), height=200)

                # FDA ranking plot
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                x2 = np.arange(3)
                crit2 = ['Side-effect', 'ADMET', 'Combined']
                sorted_se = sorted(side_effect_scores.items(), key=lambda x: x[1])
                rank_f1 = {d: idx+1 for idx, (d, _) in enumerate(sorted_se)}
                sorted_admet_fda = sorted(fda_admet_scores.items(), key=lambda x: x[1], reverse=True)
                rank_f2 = {d: idx+1 for idx, (d, _) in enumerate(sorted_admet_fda)}
                rank_f3 = {item['Drug Name']: item['Rank'] for item in fda_final}
                for d in fda_drug_list:
                    ax2.plot(x2, [rank_f1.get(d, np.nan), rank_f2.get(d, np.nan), rank_f3.get(d, np.nan)],
                            marker='o', label=d)
                ax2.set_xticks(x2)
                ax2.set_xticklabels(crit2)
                ax2.set_ylabel('Rank')
                ax2.set_title('FDA Drug Ranks by Criteria')
                ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), frameon=False)
                st.pyplot(fig2)

            # DDI and Final Pair Ranking (full display, Rank first)
            st.write("-------------------")
            st.write(" ")
            st.warning("DRUG-DRUG INTERACTION (DDI) PREDICTION")
            
            ddi_df = ddi.predict_for_all_combinations(cancer_drug_smiles_list, fda_drugs)[
                ["Cancer_Drug", "FDA_Drug", "Map1"]
            ]
            st.dataframe(ddi_df, height=200)
            st.write("-------------------")
            st.write(" ")
            st.success("FINAL COMBINED DRUG PAIR RANKING")
            
            final_pair = final_rank.final_pair_topsis(cancer_final, fda_final,
                                                    ddi.predict_for_all_combinations(cancer_drug_smiles_list, fda_drugs))
            df_final_pair = pd.DataFrame(final_pair)
            st.dataframe(reorder_rank(df_final_pair), height=200)'''


# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Imported modules from your project
import quantum_modell
import cancer_admet_module
import cancer_combined_ranking
import docking
import fda_admet
import fda_side_effects
import fda_combined_ranking
import ddi
import final_rank

# Helper to move Rank column to the front
def reorder_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "Rank" in df.columns:
        return df[["Rank"] + [c for c in df.columns if c != "Rank"]]
    return df

# Make the app use the full width
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* make sure it's only our custom links */
    a.page-link-button {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        margin: 0 0.3rem;
        border: none;
        border-radius: 8px;
        background-color: #ECECEC !important;    /* light neutral grey */
        color: #000000 !important;               /* pure black text */
        font-weight: 600;
        text-decoration: none;
        transition: background-color 0.2s ease;
    }
    a.page-link-button:hover {
        background-color: #CCCCCC !important;    /* slightly darker on hover */
        color: #000000 !important;
    }
    /* ensure streamlit buttons aren't squashed */
    .stButton>button {
        height: auto;
    }
    /* tighten up container padding */
    div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Top banner
st.title("💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 💉 🧬 ")
st.write(" ")
st.title(" Drug Ranking System for Co‑Morbid Patients")
st.write("")

# Navigation row — point at root‑level slugs, not pages/… paths
cols = st.columns(5)
pages = [
    ("Quantum",    "Quantum"),
    ("ADMET",      "ADMET"),
    ("SE",         "Side Effect Severity"),
    ("Docking_ui", "Active Torsion Sites"),
    ("DDI_ui",     "Drug‑Drug Interaction"),
]
for col, (slug, label) in zip(cols, pages):
    with col:
        # link to "/Quantum", "/ADMET", etc.
        st.markdown(
            f'<a href="/{slug}" class="page-link-button">{label}</a>',
            unsafe_allow_html=True
        )

st.write("---")

# Main analysis form
st.header("Run Combined Cancer & FDA Analysis")
with st.form(key="analysis_form"):
    protein_input = st.text_input("Enter protein sequence:")
    n_drugs       = st.number_input("Number of drugs to consider:", min_value=1, step=1)
    disease       = st.text_input("Enter a disease name:")
    submit_button = st.form_submit_button("Run Analysis")

if submit_button:
    with st.spinner("Analyzing…"):
        # === Cancer Analysis ===
        kiba_df = pd.read_csv(r"C:\Users\sthdh\Downloads\KIBA.csv\KIBA.csv")
        kiba_df = kiba_df[kiba_df["target_sequence"]
                            .str.contains(protein_input, case=False, na=False)]
        if kiba_df.empty:
            st.error(f"No drugs found for protein '{protein_input}'.")
        else:
            # 1) Quantum binding affinities
            drug_smiles_list = (
                kiba_df["compound_iso_smiles"]
                .dropna().unique().tolist()
            )
            preds = quantum_modell.predict_binding_affinities(
                        drug_smiles_list, protein_input)
            preds_sorted = sorted(
                        preds,
                        key=lambda x: x["predicted_affinity"],
                        reverse=True)
            top_n_cancer = preds_sorted[:n_drugs]
            for i, e in enumerate(top_n_cancer):
                e["Rank"] = i+1
            bind_dict = {e["drug"]: e["predicted_affinity"] 
                         for e in top_n_cancer}

            # 2) Docking
            dock_results = docking.perform_docking(
                                [e["drug"] for e in top_n_cancer],
                                protein_input)
            dock_results = sorted(dock_results, 
                                  key=lambda x: x["sites"], 
                                  reverse=True)
            prev, rank = None, 0
            for item in dock_results:
                if item["sites"] != prev:
                    rank += 1
                    prev = item["sites"]
                item["Rank"] = rank
            dock_dict = {e["drug"]: e["sites"] for e in dock_results}

            # 3) Cancer ADMET
            cadmet = cancer_admet_module.cancer_admet_analysis(
                         [e["drug"] for e in top_n_cancer], flag=0)
            cadmet_df = (
                pd.DataFrame(cadmet.items(), 
                             columns=["Drug","ADMET Score"])
                  .sort_values("ADMET Score", ascending=False)
                  .reset_index(drop=True)
            )
            cadmet_df.insert(0,"Rank", range(1,len(cadmet_df)+1))

            # 4) Cancer TOPSIS
            cancer_final = cancer_combined_ranking.cancer_final_combined_topsis(
                               bind_dict, cadmet, dock_dict)
            for i,e in enumerate(cancer_final):
                e["Rank"] = i+1

            # === FDA Analysis ===
            side_effects, fda_drugs = (
                fda_side_effects.fetch_drug_info(disease, 0)
            )
            se_scores = {
                i["Generic Name"]: i["Normalized Cumulative Score"]
                for i in side_effects 
                if "Normalized Cumulative Score" in i
            }
            fda_admet_scores = fda_admet.fda_admet_analysis(fda_drugs, 0)
            fda_final = fda_combined_ranking.fda_final_combined_topsis(
                            se_scores, fda_admet_scores)
            for i,e in enumerate(fda_final):
                e["Rank"] = i+1

            fda_admet_df = pd.DataFrame(
                fda_admet_scores.items(),
                columns=["Drug","ADMET Score"]
            )
            fda_admet_df.insert(0,"Rank", range(1,len(fda_admet_df)+1))

            # === Layout ===
            c1, c_div, c2 = st.columns([5,0.5,5])
            with c1:
                st.subheader("Cancer: Binding Affinity")
                st.dataframe(reorder_rank(pd.DataFrame(top_n_cancer)), height=200)
                st.subheader("Cancer: Docking Sites")
                st.dataframe(reorder_rank(pd.DataFrame(dock_results)), height=200)
                st.subheader("Cancer: ADMET")
                st.dataframe(cadmet_df, height=200)
                st.subheader("Cancer: Combined TOPSIS")
                st.dataframe(reorder_rank(pd.DataFrame(cancer_final)), height=200)

            with c_div:
                st.markdown(
                  "<div style='border-left:2px solid black; height:100%;'></div>",
                  unsafe_allow_html=True
                )

            with c2:
                st.subheader("FDA: Side‑Effect Severity")
                st.dataframe(pd.DataFrame(side_effects), height=200)
                st.subheader("FDA: ADMET Scores")
                st.dataframe(reorder_rank(fda_admet_df), height=200)
                st.subheader("FDA: Combined TOPSIS")
                st.dataframe(
                    reorder_rank(pd.DataFrame(fda_final)),
                    height=200
                )

            # DDI + Final Pair
            st.warning("Drug‑Drug Interaction Predictions")
            ddi_df = ddi.predict_for_all_combinations(
                         [e["drug"] for e in top_n_cancer],
                         fda_drugs
                     )[[ "Cancer_Drug", "FDA_Drug", "Map1" ]]
            st.dataframe(ddi_df, height=200)

            st.success("FINAL COMBINED DRUG PAIR RANKING")
            final_pair = final_rank.final_pair_topsis(
                cancer_final, fda_final,
                ddi.predict_for_all_combinations(
                    [e["drug"] for e in top_n_cancer], fda_drugs
                )
            )
            st.dataframe(reorder_rank(pd.DataFrame(final_pair)), height=200)




