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
