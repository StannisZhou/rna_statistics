import os
import pickle

import data
import results

K, r = 10000, 4
data_folder = os.path.abspath('data')
raw_data_fname = 'intermediate/raw_data.pkl'
processed_data_fname = 'intermediate/rna_molecules.pkl'
markov_shuffles_fname = 'intermediate/markov_shuffles.pkl'
if not os.path.exists('intermediate'):
    os.mkdir('intermediate')

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

essential_data = data.get_essential_data(rna_molecules, markov_shuffles)
essential_data_by_family = results.get_essential_data_by_family(essential_data)

exploratory_comparative_fname = 'exploratory_comparative.tex'
exploratory_mfe_fname = 'exploratory_mfe.tex'
bound_unbound_fig_fname = 'bound_unbound.png'
bound_unbound_tex_fname = 'bound_unbound.tex'
comparative_mfe_fig_fname = 'comparative_mfe.png'
comparative_mfe_tex_fname = 'comparative_mfe.tex'
formal_analysis_fname = 'formal_analysis.tex'
data_summary_fname = 'data_summary.tex'

results.generate_exploratory_analysis_tables(
    essential_data_by_family, exploratory_comparative_fname, exploratory_mfe_fname
)
results.generate_formal_analysis_table(essential_data_by_family, formal_analysis_fname)
results.generate_bound_unbound_figure(
    essential_data_by_family, bound_unbound_fig_fname, bound_unbound_tex_fname
)
results.generate_comparative_mfe_figure(
    essential_data_by_family, comparative_mfe_fig_fname, comparative_mfe_tex_fname
)
results.generate_data_summary_table(essential_data_by_family, data_summary_fname)
