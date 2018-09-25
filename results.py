import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rna
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import hypergeom

GROUP_NAMES = {
    'Group_I_Introns': 'Group I Introns',
    'Group_II_Introns': 'Group II Introns',
    'tmRNAs': 'tmRNA',
    'SRP_RNAs': 'SRP RNA',
    'RNase_Ps': 'RNase P',
    '16s_ribosomal': '16s rRNA',
    '23s_ribosomal': '23s rRNA'
}
SINGLE_ENTITY_RNAS = ['Group_I_Introns', 'Group_II_Introns']
PROTEIN_RNA_COMPLEXES = [
    'SRP_RNAs', 'tmRNAs', 'RNase_Ps', '16s_ribosomal', '23s_ribosomal'
]


def get_essential_data_by_family(essential_data):
    essential_data_by_family = {}
    for group in essential_data:
        n_molecules = len(list(essential_data[group].keys()))
        molecule_lengths = np.zeros(n_molecules, dtype=int)
        ambiguity_index = {
            'comparative': {
                'T-S': np.zeros(n_molecules),
                'D-S': np.zeros(n_molecules)
            },
            'mfe': {
                'T-S': np.zeros(n_molecules),
                'D-S': np.zeros(n_molecules)
            }
        }
        alpha_index = {
            'comparative': {
                'T-S': np.zeros(n_molecules),
                'D-S': np.zeros(n_molecules)
            },
            'mfe': {
                'T-S': np.zeros(n_molecules),
                'D-S': np.zeros(n_molecules)
            }
        }
        for ff, fname in enumerate(essential_data[group]):
            molecule_lengths[ff] = essential_data[group][fname]['length']
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
            'ambiguity_index': ambiguity_index,
            'alpha_index': alpha_index
        }

    return essential_data_by_family


def generate_exploratory_analysis_tables(essential_data_by_family, comparative_fname, mfe_fname):
    with open(os.path.join('paper', comparative_fname), 'w') as f:
        for group in rna.RNA_GROUPS:
            data = essential_data_by_family[group]
            f.write(' {} & {} & {} & {:.3f} & {:.3f} \\\\ \\hline\n'.format(
                GROUP_NAMES[group], data['n_molecules'],
                int(np.median(data['molecule_lengths'])),
                np.median(data['alpha_index']['comparative']['T-S']),
                np.median(data['alpha_index']['comparative']['D-S'])
            ))

    with open(os.path.join('paper', mfe_fname), 'w') as f:
        for group in rna.RNA_GROUPS:
            data = essential_data_by_family[group]
            f.write(' {} & {} & {} & {:.3f} & {:.3f} \\\\ \\hline\n'.format(
                GROUP_NAMES[group], data['n_molecules'],
                int(np.median(data['molecule_lengths'])),
                np.median(data['alpha_index']['mfe']['T-S']),
                np.median(data['alpha_index']['mfe']['D-S'])
            ))


def generate_bound_unbound_figure(essential_data_by_family, fig_fname, tex_fname):
    ambiguity_index = {
        'comparative': {
            'T-S': [],
            'D-S': []
        },
        'mfe': {
            'T-S': [],
            'D-S': []
        }
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
            auc_score = roc_auc_score(
                labels, -ambiguity_index[key][index]
            )
            p_value = get_hypergeometric_p_value(
                ambiguity_index[key][index], labels
            )
            results[key][index] = {
                'true_positive_rates': true_positive_rates,
                'false_positive_rates': false_positive_rates,
                'auc_score': auc_score,
                'p_value': p_value
            }

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(2, 2)
    for ii, index in enumerate(['T-S', 'D-S']):
        for jj, key in enumerate(['comparative', 'mfe']):
            x = results[key][index]['false_positive_rates']
            y = results[key][index]['true_positive_rates']
            auc = results[key][index]['auc_score']
            ax[ii, jj].plot(x, y)
            ax[ii, jj].set_xlabel('false positive rate')
            ax[ii, jj].set_ylabel('true positive rate')
            if key == 'comparative':
                name = key.capitalize()
            else:
                name = key.upper()

            ax[ii, jj].set_title('{}, $d_{{ {} }} < t$'.format(name, index))
            ax[ii, jj].text(0.6, 0.2, 'AUC = {:.2f}'.format(auc))

    fig.tight_layout()
    fig.savefig(os.path.join('paper', fig_fname), dpi=400)
    with open(os.path.join('paper', tex_fname), 'w') as f:
        f.write('\\begin{figure}[h!]\n')
        f.write('\\centering\n')
        f.write('\\includegraphics[width=0.7\\textwidth]{' + fig_fname +'}\n')
        f.write('\\vglue 0.5cm\n')
        f.write('''
\\caption{{{{\\bf Bound or Unbound?}} ROC performance of classifiers based on thresholding T-S
and D-S ambiguity indexes. Small values are taken as evidence for molecules that are active as
single entities (unbound), as opposed to parts of ribonucleoproteins (bound). Classifiers in the
left two panels use comparative secondary structures to compute ambiguity indexes; those on the
right use (approximate) minimum free energies. In each of the four experiments, a conditional
p-value was also calculated, based only on the signs of the indexes and the null hypothesis that
positive indexes are distributed randomly among molecules of all types as opposed to the alternative
that positive indexes are more typically found among families of bound RNA. Under the null hypothesis,
the test statistic is hypergeometric---see Eq \\ref{{eqn:null}}. {{\\em Upper Left:}} $p= {} $;
{{\em Lower Left:}} $p={}$; {{\\em Upper Right:}} $p={:.2f}$;  {{\\em Lower Right:}} $p={:.2f}$.}}
'''.format(latex_float(results['comparative']['T-S']['p_value']), latex_float(results['comparative']['D-S']['p_value']), results['mfe']['T-S']['p_value'], results['mfe']['D-S']['p_value']))
        f.write('\\label{fig:UnboundVSBound}\n')
        f.write('\\end{figure}\n')


def generate_comparative_mfe_figure(essential_data_by_family, fig_fname, tex_fname):
    ambiguity_index = {
        'bound': {
            'T-S': [],
            'D-S': []
        },
        'unbound': {
            'T-S': [],
            'D-S': []
        }
    }
    labels = {
        'bound': [],
        'unbound': []
    }
    for key in ['comparative', 'mfe']:
        for group in SINGLE_ENTITY_RNAS:
            for index in ambiguity_index['unbound']:
                ambiguity_index['unbound'][index].append(
                    essential_data_by_family[group]['ambiguity_index'][key][index]
                )

            if key == 'comparative':
                labels['unbound'].append(np.ones(essential_data_by_family[group]['n_molecules']))
            elif key == 'mfe':
                labels['unbound'].append(np.zeros(essential_data_by_family[group]['n_molecules']))
            else:
                raise Exception('Unknown secondary structure {}'.format(group))

        for group in PROTEIN_RNA_COMPLEXES:
            for index in ambiguity_index['bound']:
                ambiguity_index['bound'][index].append(
                    essential_data_by_family[group]['ambiguity_index'][key][index]
                )

            if key == 'comparative':
                labels['bound'].append(np.ones(essential_data_by_family[group]['n_molecules']))
            elif key == 'mfe':
                labels['bound'].append(np.zeros(essential_data_by_family[group]['n_molecules']))
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
            auc_score = roc_auc_score(
                labels[key], -ambiguity_index[key][index]
            )
            p_value = get_hypergeometric_p_value(
                ambiguity_index[key][index], labels[key]
            )
            results[key][index] = {
                'true_positive_rates': true_positive_rates,
                'false_positive_rates': false_positive_rates,
                'auc_score': auc_score,
                'p_value': p_value
            }

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(2, 2)
    for ii, index in enumerate(['T-S', 'D-S']):
        for jj, key in enumerate(['unbound', 'bound']):
            x = results[key][index]['false_positive_rates']
            y = results[key][index]['true_positive_rates']
            auc = results[key][index]['auc_score']
            ax[ii, jj].plot(x, y)
            ax[ii, jj].set_xlabel('false positive rate')
            ax[ii, jj].set_ylabel('true positive rate')
            ax[ii, jj].set_title('{} RNA, $d_{{ {} }} < t$'.format(key.capitalize(), index))
            ax[ii, jj].text(0.6, 0.2, 'AUC = {:.2f}'.format(auc))

    fig.tight_layout()
    fig.savefig(os.path.join('paper', fig_fname), dpi=400)
    with open(os.path.join('paper', tex_fname), 'w') as f:
        f.write('\\begin{figure}[h!]\n')
        f.write('\\centering\n')
        f.write('\\includegraphics[width=0.7\\textwidth]{' + fig_fname + '}\n')
        f.write('\\vglue 0.5cm\n')
        f.write('''
\\caption{{{{\\bf Comparative or MFE?}} As in Figure \\ref{{fig:UnboundVSBound}}, each panel
depicts the ROC performance of a classifier based on thresholding the T-S (top two panels) or
D-S (bottom two panels) ambiguity indexes. Here, small values are taken as evidence for
comparative as opposed to MFE secondary structure. Either index, T-S or D-S, can be used to
construct a good classifier of the origin of a secondary structure for the unbound molecules in
our data set (left two panels) but not for the bound molecules (right two panels). Conditional
p-values were also calculated, using the hypergeometric distribution and based only on the signs
of the indexes. In each case and the null hypothesis is that comparative secondary structures are as
likely to lead to positive ambiguity indexes as are MFE structures, whereas the alternative is
that positive ambiguity indexes are more typical when derived from MFE structures:
{{\\em Upper Left:}} $p= {} $; {{\em Upper Right:}} $p={:.2f}$; {{\\em Lower Left:}} $p={}$;  {{\\em Lower Right:}} $p={:.2f}$.}}
'''.format(latex_float(results['unbound']['T-S']['p_value']), results['bound']['T-S']['p_value'], latex_float(results['unbound']['D-S']['p_value']), results['bound']['D-S']['p_value']))
        f.write('\\label{fig:CompVSMFE}\n')
        f.write('\\end{figure}\n')


def generate_formal_analysis_table(essential_data_by_family, fname):
    with open(os.path.join('paper', fname), 'w') as f:
        for group in rna.RNA_GROUPS:
            data = essential_data_by_family[group]
            f.write(' {} & {} & {} & {} & {} & {} \\\\  \\hline\n'.format(
                GROUP_NAMES[group], data['n_molecules'],
                np.sum(data['ambiguity_index']['comparative']['T-S'] > 0),
                np.sum(data['ambiguity_index']['comparative']['D-S'] > 0),
                np.sum(data['ambiguity_index']['mfe']['T-S'] > 0),
                np.sum(data['ambiguity_index']['mfe']['D-S'] > 0),
            ))


def generate_data_summary_table(essential_data_by_family, fname):
    with open(os.path.join('paper', fname), 'w') as f:
        for group in rna.RNA_GROUPS:
            data = essential_data_by_family[group]
            f.write(' {} & {} & {} & {} & {} \\\\ \\hline\n'.format(
                GROUP_NAMES[group], data['n_molecules'],
                np.min(data['molecule_lengths']),
                np.max(data['molecule_lengths']),
                int(np.median(data['molecule_lengths']))
            ))


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
