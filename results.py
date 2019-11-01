from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

GROUP_NAMES = {
    'Group_I_Introns': 'Group I Introns',
    'Group_II_Introns': 'Group II Introns',
    'tmRNAs': 'tmRNA',
    'SRP_RNAs': 'SRP RNA',
    'RNase_Ps': 'RNase P',
    '16s_ribosomal': '16s rRNA',
    '23s_ribosomal': '23s rRNA',
    'Group_I_Introns_A1': 'Group I Introns A1',
    'Group_I_Introns_A2': 'Group I Introns A2',
    'Group_I_Introns_A3': 'Group I Introns A3',
    'Group_I_Introns_B1': 'Group I Introns B1',
    'Group_I_Introns_B2': 'Group I Introns B2',
    'Group_I_Introns_B3': 'Group I Introns B3',
    'Group_I_Introns_B4': 'Group I Introns B4',
    'Group_I_Introns_C1': 'Group I Introns C1',
    'Group_I_Introns_C2': 'Group I Introns C2',
    'Group_I_Introns_C3': 'Group I Introns C3',
    'Group_I_Introns_D': 'Group I Introns D',
    'Group_I_Introns_E': 'Group I Introns E',
    'Group_I_Introns_UNK': 'Group I Introns Unknown',
    'Group_II_Introns_A': 'Group II Introns A',
    'Group_II_Introns_B': 'Group II Introns B',
}
SINGLE_ENTITY_RNAS = ['Group_I_Introns', 'Group_II_Introns']
PROTEIN_RNA_COMPLEXES = [
    'SRP_RNAs',
    'tmRNAs',
    'RNase_Ps',
    '16s_ribosomal',
    '23s_ribosomal',
]


def get_essential_data_by_family(essential_data):
    essential_data_by_family = {}
    for group in essential_data:
        n_molecules = len(list(essential_data[group].keys()))
        molecule_lengths = np.zeros(n_molecules, dtype=int)
        n_pseudoknots = np.zeros(n_molecules, dtype=int)
        ambiguity_index = {
            'comparative': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
            'mfe': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
        }
        alpha_index = {
            'comparative': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
            'mfe': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
        }
        for ff, fname in enumerate(essential_data[group]):
            molecule_lengths[ff] = essential_data[group][fname]['length']
            n_pseudoknots[ff] = essential_data[group][fname]['n_pseudoknots']
            for key in ['comparative', 'mfe']:
                for index in ['T-S', 'D-S']:
                    ambiguity_index[key][index][ff] = essential_data[group][fname][
                        'ambiguity_index_{}_{}'.format(key, index)
                    ]
                    alpha_index[key][index][ff] = essential_data[group][fname][
                        'alpha_index_{}_{}'.format(key, index)
                    ]

        essential_data_by_family[group] = {
            'n_molecules': n_molecules,
            'molecule_lengths': molecule_lengths,
            'n_pseudoknots': n_pseudoknots,
            'ambiguity_index': ambiguity_index,
            'alpha_index': alpha_index,
        }

    return essential_data_by_family


def get_essential_data_by_family_for_intron_subclasses(essential_data, raw_data):
    # Get Group I Intron subclasses
    intron_subclasses = {
        'Group_I_Introns': defaultdict(list),
        'Group_II_Introns': defaultdict(list),
    }
    for fname in raw_data['Group_I_Introns']:
        header = raw_data['Group_I_Introns'][fname]['header']
        header = header[-1].split(', ')[1].split('.')
        rna_class = [
            component
            for component in header
            if component
            in [
                'A1',
                'A2',
                'A3',
                'B1',
                'B2',
                'B3',
                'B4',
                'C1',
                'C2',
                'C3',
                'D',
                'E',
                'UNK',
            ]
        ]
        intron_subclasses['Group_I_Introns'][rna_class[-1]].append(fname)

    # Get Group II Intron subclasses
    for fname in raw_data['Group_II_Introns']:
        header = raw_data['Group_II_Introns'][fname]['header']
        header = header[-1].split(', ')[1].split('.')
        intron_subclasses['Group_II_Introns'][header[5]].append(fname)

    essential_data_for_intron_subclasses = {}
    for group in ['Group_I_Introns', 'Group_II_Introns']:
        for subclass in intron_subclasses[group]:
            n_molecules = len(intron_subclasses[group][subclass])
            molecule_lengths = np.zeros(n_molecules, dtype=int)
            n_pseudoknots = np.zeros(n_molecules, dtype=int)
            ambiguity_index = {
                'comparative': {
                    'T-S': np.zeros(n_molecules),
                    'D-S': np.zeros(n_molecules),
                },
                'mfe': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
            }
            alpha_index = {
                'comparative': {
                    'T-S': np.zeros(n_molecules),
                    'D-S': np.zeros(n_molecules),
                },
                'mfe': {'T-S': np.zeros(n_molecules), 'D-S': np.zeros(n_molecules)},
            }
            for ff, fname in enumerate(intron_subclasses[group][subclass]):
                molecule_lengths[ff] = essential_data[group][fname]['length']
                n_pseudoknots[ff] = essential_data[group][fname]['n_pseudoknots']
                for key in ['comparative', 'mfe']:
                    for index in ['T-S', 'D-S']:
                        ambiguity_index[key][index][ff] = essential_data[group][fname][
                            'ambiguity_index_{}_{}'.format(key, index)
                        ]
                        alpha_index[key][index][ff] = essential_data[group][fname][
                            'alpha_index_{}_{}'.format(key, index)
                        ]

            essential_data_for_intron_subclasses['{}_{}'.format(group, subclass)] = {
                'n_molecules': n_molecules,
                'molecule_lengths': molecule_lengths,
                'n_pseudoknots': n_pseudoknots,
                'ambiguity_index': ambiguity_index,
                'alpha_index': alpha_index,
            }

    return essential_data_for_intron_subclasses


def generate_exploratory_analysis_tables(essential_data_by_family):
    comparative_table = {
        'family': [GROUP_NAMES[group] for group in essential_data_by_family],
        'number molecules': [
            essential_data_by_family[group]['n_molecules']
            for group in essential_data_by_family
        ],
        'median length': [
            int(np.median(essential_data_by_family[group]['molecule_lengths']))
            for group in essential_data_by_family
        ],
        'median alpha_{T-S}': [
            np.median(
                essential_data_by_family[group]['alpha_index']['comparative']['T-S']
            )
            for group in essential_data_by_family
        ],
        'median alpha_{D-S}': [
            np.median(
                essential_data_by_family[group]['alpha_index']['comparative']['D-S']
            )
            for group in essential_data_by_family
        ],
    }
    mfe_table = {
        'family': [GROUP_NAMES[group] for group in essential_data_by_family],
        'number molecules': [
            essential_data_by_family[group]['n_molecules']
            for group in essential_data_by_family
        ],
        'median length': [
            int(np.median(essential_data_by_family[group]['molecule_lengths']))
            for group in essential_data_by_family
        ],
        'median alpha_{T-S}': [
            np.median(essential_data_by_family[group]['alpha_index']['mfe']['T-S'])
            for group in essential_data_by_family
        ],
        'median alpha_{D-S}': [
            np.median(essential_data_by_family[group]['alpha_index']['mfe']['D-S'])
            for group in essential_data_by_family
        ],
    }
    return pd.DataFrame(data=comparative_table), pd.DataFrame(data=mfe_table)


def generate_bound_unbound_figure(essential_data_by_family):
    ambiguity_index = {
        'comparative': {'T-S': [], 'D-S': []},
        'mfe': {'T-S': [], 'D-S': []},
    }
    labels = []
    for group in essential_data_by_family:
        for key in ambiguity_index:
            for index in ambiguity_index[key]:
                ambiguity_index[key][index].append(
                    essential_data_by_family[group]['ambiguity_index'][key][index]
                )

        if group in SINGLE_ENTITY_RNAS:
            labels.append(np.ones(essential_data_by_family[group]['n_molecules']))
        elif group in PROTEIN_RNA_COMPLEXES:
            labels.append(np.zeros(essential_data_by_family[group]['n_molecules']))
        else:
            raise Exception('Unknown group {}'.format(group))

    for key in ambiguity_index:
        for index in ambiguity_index[key]:
            ambiguity_index[key][index] = np.concatenate(ambiguity_index[key][index])

    labels = np.concatenate(labels)
    results = {}
    for key in ambiguity_index:
        results[key] = {}
        for index in ambiguity_index[key]:
            false_positive_rates, true_positive_rates, _ = roc_curve(
                labels, -ambiguity_index[key][index]
            )
            auc_score = roc_auc_score(labels, -ambiguity_index[key][index])
            p_value = get_hypergeometric_p_value(ambiguity_index[key][index], labels)
            results[key][index] = {
                'true_positive_rates': true_positive_rates,
                'false_positive_rates': false_positive_rates,
                'auc_score': auc_score,
                'p_value': p_value,
            }

    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(2, 2)
    for ii, index in enumerate(['T-S', 'D-S']):
        for jj, key in enumerate(['comparative', 'mfe']):
            x = results[key][index]['false_positive_rates']
            y = results[key][index]['true_positive_rates']
            y = lowess(y, x, 0.1, return_sorted=False)
            auc = results[key][index]['auc_score']
            p_value = latex_float(results[key][index]['p_value'])
            ax[ii, jj].plot(x, y, linewidth=2.0)
            ax[ii, jj].set_xlabel('false positive rate', fontweight='bold')
            ax[ii, jj].set_ylabel('true positive rate', fontweight='bold')
            if key == 'comparative':
                name = key.capitalize()
            else:
                name = key.upper()

            ax[ii, jj].set_title(
                '{}, $d_{{ {} }} < t$, p value ${}$'.format(name, index, p_value),
                fontweight='bold',
            )
            ax[ii, jj].text(0.6, 0.2, 'AUC = {:.2f}'.format(auc), fontweight='bold')

    fig.tight_layout()
    return fig


def generate_comparative_mfe_figure(essential_data_by_family):
    ambiguity_index = {
        'bound': {'T-S': [], 'D-S': []},
        'unbound': {'T-S': [], 'D-S': []},
    }
    labels = {'bound': [], 'unbound': []}
    for key in ['comparative', 'mfe']:
        for group in SINGLE_ENTITY_RNAS:
            for index in ambiguity_index['unbound']:
                ambiguity_index['unbound'][index].append(
                    essential_data_by_family[group]['ambiguity_index'][key][index]
                )

            if key == 'comparative':
                labels['unbound'].append(
                    np.ones(essential_data_by_family[group]['n_molecules'])
                )
            elif key == 'mfe':
                labels['unbound'].append(
                    np.zeros(essential_data_by_family[group]['n_molecules'])
                )
            else:
                raise Exception('Unknown secondary structure {}'.format(group))

        for group in PROTEIN_RNA_COMPLEXES:
            for index in ambiguity_index['bound']:
                ambiguity_index['bound'][index].append(
                    essential_data_by_family[group]['ambiguity_index'][key][index]
                )

            if key == 'comparative':
                labels['bound'].append(
                    np.ones(essential_data_by_family[group]['n_molecules'])
                )
            elif key == 'mfe':
                labels['bound'].append(
                    np.zeros(essential_data_by_family[group]['n_molecules'])
                )
            else:
                raise Exception('Unknown secondary structure {}'.format(group))

    for key in ambiguity_index:
        for index in ambiguity_index[key]:
            ambiguity_index[key][index] = np.concatenate(ambiguity_index[key][index])

    for key in labels:
        labels[key] = np.concatenate(labels[key])

    results = {}
    for key in ambiguity_index:
        results[key] = {}
        for index in ambiguity_index[key]:
            false_positive_rates, true_positive_rates, _ = roc_curve(
                labels[key], -ambiguity_index[key][index]
            )
            auc_score = roc_auc_score(labels[key], -ambiguity_index[key][index])
            p_value = get_hypergeometric_p_value(
                ambiguity_index[key][index], labels[key]
            )
            results[key][index] = {
                'true_positive_rates': true_positive_rates,
                'false_positive_rates': false_positive_rates,
                'auc_score': auc_score,
                'p_value': p_value,
            }

    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(2, 2)
    for ii, index in enumerate(['T-S', 'D-S']):
        for jj, key in enumerate(['unbound', 'bound']):
            x = results[key][index]['false_positive_rates']
            y = results[key][index]['true_positive_rates']
            y = lowess(y, x, 0.1, return_sorted=False)
            auc = results[key][index]['auc_score']
            p_value = latex_float(results[key][index]['p_value'])
            ax[ii, jj].plot(x, y, linewidth=2.0)
            ax[ii, jj].set_xlabel('false positive rate', fontweight='bold')
            ax[ii, jj].set_ylabel('true positive rate', fontweight='bold')
            ax[ii, jj].set_title(
                '{} RNA, $d_{{ {} }} < t$, p value ${}$'.format(
                    key.capitalize(), index, p_value
                ),
                fontweight='bold',
            )
            ax[ii, jj].text(0.6, 0.2, 'AUC = {:.2f}'.format(auc), fontweight='bold')

    fig.tight_layout()
    return fig


def generate_formal_analysis_table(essential_data_by_family):
    comparative_table = {
        'family': [GROUP_NAMES[group] for group in essential_data_by_family],
        '# of molecules': [
            essential_data_by_family[group]['n_molecules']
            for group in essential_data_by_family
        ],
        '# of posotive T-S indexes': [
            np.sum(
                essential_data_by_family[group]['ambiguity_index']['comparative']['T-S']
                > 0
            )
            for group in essential_data_by_family
        ],
        '# of positive D-S indexes': [
            np.sum(
                essential_data_by_family[group]['ambiguity_index']['comparative']['D-S']
                > 0
            )
            for group in essential_data_by_family
        ],
    }
    mfe_table = {
        'family': [GROUP_NAMES[group] for group in essential_data_by_family],
        '# of molecules': [
            essential_data_by_family[group]['n_molecules']
            for group in essential_data_by_family
        ],
        '# of positive T-S indexes': [
            np.sum(essential_data_by_family[group]['ambiguity_index']['mfe']['T-S'] > 0)
            for group in essential_data_by_family
        ],
        '# of positive D-S indexes': [
            np.sum(essential_data_by_family[group]['ambiguity_index']['mfe']['D-S'] > 0)
            for group in essential_data_by_family
        ],
    }
    return pd.DataFrame(data=comparative_table), pd.DataFrame(data=mfe_table)


def generate_data_summary_table(essential_data_by_family):
    table = {
        'family': [GROUP_NAMES[group] for group in essential_data_by_family],
        'number': [
            essential_data_by_family[group]['n_molecules']
            for group in essential_data_by_family
        ],
        'number w/ pseudoknots': [
            np.sum(essential_data_by_family[group]['n_pseudoknots'] > 0)
            for group in essential_data_by_family
        ],
        'min length': [
            np.min(essential_data_by_family[group]['molecule_lengths'])
            for group in essential_data_by_family
        ],
        'max length': [
            np.max(essential_data_by_family[group]['molecule_lengths'])
            for group in essential_data_by_family
        ],
        'median length': [
            np.median(essential_data_by_family[group]['molecule_lengths'])
            for group in essential_data_by_family
        ],
    }
    return pd.DataFrame(data=table)


def get_hypergeometric_p_value(values, labels):
    M = len(values)
    n = np.sum(values > 0)
    N = np.sum(labels)
    x = np.sum(values[labels == 1] > 0)
    return hypergeom.cdf(x, M, n, N)


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
