import os
import pickle

import data
import results
import streamlit as st

K, r = 10000, 4
data_folder = os.path.abspath('data')
raw_data_fname = 'intermediate/raw_data.pkl'
processed_data_fname = 'intermediate/rna_molecules.pkl'
markov_shuffles_fname = 'intermediate/markov_shuffles.pkl'
if not os.path.exists('intermediate'):
    os.mkdir('intermediate')

# Run all experiments and cache the results
if not os.path.exists(raw_data_fname):
    rna_raw_data = data.process_raw_data(data_folder)
    with open(raw_data_fname, 'wb') as f:
        pickle.dump(rna_raw_data, f)
else:
    with open(raw_data_fname, 'rb') as f:
        rna_raw_data = pickle.load(f)

if not os.path.exists(processed_data_fname):
    rna_molecules = data.get_rna_molecules(rna_raw_data, K, r)
    with open(processed_data_fname, 'wb') as f:
        pickle.dump(rna_molecules, f)
else:
    with open(processed_data_fname, 'rb') as f:
        rna_molecules = pickle.load(f)

if not os.path.exists(markov_shuffles_fname):
    markov_shuffles = data.get_markov_shuffles(rna_molecules)
    with open(markov_shuffles_fname, 'wb') as f:
        pickle.dump(markov_shuffles, f)
else:
    with open(markov_shuffles_fname, 'rb') as f:
        markov_shuffles = pickle.load(f)

st.title(
    'Supplementary Results for *Base-pair Ambiguity and the Kinetics of RNA Folding*'
)
for use_pseudoknot_free in [False, True]:
    essential_data = data.get_essential_data(
        rna_molecules, markov_shuffles, use_pseudoknot_free=use_pseudoknot_free
    )
    essential_data_by_family = results.get_essential_data_by_family(essential_data)
    if not use_pseudoknot_free:
        st.header('Data Summary Table')
        data_summary_table = results.generate_data_summary_table(
            essential_data_by_family
        )
        st.write(data_summary_table)
        st.header('Results with Original Comparative Structures')
    else:
        st.header('Results with Pseudoknot-free Comparative Structures')

    comparative_table, mfe_table = results.generate_exploratory_analysis_tables(
        essential_data_by_family
    )

    st.subheader('Comparative Structures: Calibrated ambiguity indexes, by RNA family')
    st.write(comparative_table)

    st.subheader('MFE Structures: Calibrated ambiguity indexes, by RNA family')
    st.write(mfe_table)

    formal_analysis_comparative_table, formal_analysis_mfe_table = results.generate_formal_analysis_table(
        essential_data_by_family
    )
    st.subheader(
        'Comparative Structures: Numbers of Positive Ambiguity Indexes, by family'
    )
    st.write(formal_analysis_comparative_table)
    st.subheader('MFE Structures: Numbers of Positive Ambiguity Indexes, by family')
    st.write(formal_analysis_mfe_table)

    st.subheader('Bound or Unbound?')
    st.write(
        'ROC performance of classifiers based on thresholding T-S and D-S ambiguity indexes'
    )
    bound_unbound_figure = results.generate_bound_unbound_figure(
        essential_data_by_family
    )
    st.write(bound_unbound_figure)

    st.subheader('Comparative or MFE?')
    st.write(
        'RCO performance of a classifier based on threshold the T-S (top two panels) or D-S (bottom two panels) ambiguity indexes'
    )
    comparative_mfe_figure = results.generate_comparative_mfe_figure(
        essential_data_by_family
    )
    st.write(comparative_mfe_figure)
