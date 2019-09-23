import multiprocessing as mp
import os
import re
import subprocess
import sys

import numpy as np

import rna
from joblib import Parallel, delayed
from tqdm import tqdm

ALPHABET = ['A', 'G', 'C', 'U']
RNA_ANALYSER_PATH = os.path.expanduser('~/RNAAnalyser/Analyser')
DATA_FOLDER = 'data'
DATA_WITHOUT_PSEUDOKNOTS_FOLDER = 'data_without_pseudoknots'


def get_n_pseudoknots(fname):
    output = str(subprocess.check_output([RNA_ANALYSER_PATH, '-analyse', fname]))
    output_lines = output.split('\\n')
    for line in output_lines:
        characters = line.split(' ')
        if len(characters) >= 3:
            if characters[0] == 'Pseudoknot' and characters[1] == 'total':
                n_pseudoknots = int(characters[2])
                break

    return n_pseudoknots


def parse_bpseq(fname):
    with open(fname) as f:
        s = f.read()
        s = s.split('\n')
        s = [x.strip() for x in s]

    print('\nbpseq file {} loaded.'.format(os.path.basename(fname)))
    sys.stderr.flush()
    primary_sequence = []
    secondary_structure = []
    header = []
    for ii in range(len(s)):
        if s[ii].startswith('1'):
            startingpoint = ii
            break

    assert s[startingpoint][0] == '1', s[startingpoint]

    for nucleotide in s[startingpoint:]:
        nucleotide = re.split('\s+', nucleotide)
        if len(nucleotide) == 1:
            break
        primary_sequence.append(nucleotide[1].upper())
        secondary_structure.append(int(nucleotide[2]) - 1)

    primary_sequence = np.array(primary_sequence)
    secondary_structure = np.array(secondary_structure)
    return header, primary_sequence, secondary_structure


def process_raw_data(data_folder):
    rna_molecules = {}
    for dirpath, dirnames, filenames in os.walk(data_folder):
        if len(filenames) > 0:
            group = os.path.basename(os.path.normpath(dirpath))
            if group in rna.RNA_GROUPS:
                rna_molecules[group] = {}

            for name in filenames:
                if name == '.DS_Store':
                    continue

                file_name = os.path.join(dirpath, name)
                print('Processing {}'.format(file_name))
                _, file_ext = os.path.splitext(file_name)
                if file_ext == '.bpseq':
                    header, primary_structure, secondary_structure = parse_bpseq(
                        file_name
                    )
                else:
                    raise Exception('Unsupported RNA data format {}'.format(file_ext))

                if sum(
                    [(nucleotide in ALPHABET) for nucleotide in primary_structure]
                ) < len(primary_structure):
                    continue

                if np.sum(secondary_structure >= 0) == 0:
                    continue

                if group in rna.RNA_GROUPS:
                    rna_molecules[group][name] = {}
                    rna_molecules[group][name]['full_path'] = file_name
                    rna_molecules[group][name]['header'] = header
                    rna_molecules[group][name]['primary_structure'] = primary_structure
                    rna_molecules[group][name][
                        'secondary_structure'
                    ] = secondary_structure

    return rna_molecules


def get_rna_molecules(rna_raw_data, K, r):
    # Generate all MFE secondary structures
    secondary_structure_mfe_outputs = Parallel(n_jobs=mp.cpu_count())(
        delayed(rna.get_vienna_rna_secondary_structures)(group, molecule, rna_raw_data)
        for group in rna_raw_data
        for molecule in rna_raw_data[group]
    )
    secondary_structure_mfe = {group: {} for group in rna_raw_data}
    for output in secondary_structure_mfe_outputs:
        secondary_structure_mfe[output[0]][output[1]] = {}
        secondary_structure_mfe[output[0]][output[1]]['mfe'] = output[2]
        if output[3] is not None:
            secondary_structure_mfe[output[0]][output[1]][
                'mfe_with_pseudoknots'
            ] = output[3]

    # Construct RNA molecules
    rna_molecules = {}
    for group in rna_raw_data:
        rna_molecules[group] = {}
        for fname in tqdm(rna_raw_data[group]):
            n_pseudoknots = get_n_pseudoknots(os.path.join(DATA_FOLDER, group, fname))
            without_pseudoknots_fname = os.path.join(
                DATA_WITHOUT_PSEUDOKNOTS_FOLDER, group, fname
            )
            secondary_structure_comparative = {
                'comparative': rna_raw_data[group][fname]['secondary_structure']
            }
            if os.path.exists(without_pseudoknots_fname):
                assert get_n_pseudoknots(without_pseudoknots_fname) == 0
                _, _, secondary_structure_without_pseudoknots = parse_bpseq(
                    without_pseudoknots_fname
                )
                secondary_structure_comparative[
                    'comparative_without_pseudoknots'
                ] = secondary_structure_without_pseudoknots

            molecule = rna.RNA(
                group=group,
                primary_structure=rna_raw_data[group][fname]['primary_structure'],
                n_pseudoknots=n_pseudoknots,
                secondary_structure_comparative=secondary_structure_comparative,
                secondary_structure_mfe=secondary_structure_mfe[group][fname],
                case='wobble',
                K=K,
                r=r,
            )
            molecule.evaluate_local_ambiguities()
            molecule.label_locations_using_secondary_structures()
            region_length = []
            for region in molecule.masks:
                for key in molecule.masks[region]:
                    region_length.append(np.sum(molecule.masks[region][key]))

            region_length = np.array(region_length)
            if np.sum(region_length > 0) == len(region_length):
                rna_molecules[group][fname] = molecule

    return rna_molecules


def get_markov_shuffles(rna_molecules, parallel=True):
    markov_shuffles = {}
    for group in rna_molecules:
        markov_shuffles[group] = {}
        for fname in tqdm(rna_molecules[group]):
            markov_shuffles[group][fname] = {}
            molecule = rna_molecules[group][fname]
            shuffles = rna.get_markov_shuffles(
                molecule.primary_structure,
                molecule.case,
                molecule.K,
                molecule.r,
                parallel,
            )
            flag = True
            for key in molecule.secondary_structure.keys():
                for index in ['T-S', 'D-S']:
                    alpha_index = molecule.get_alpha_index(key, index, shuffles)
                    if np.isnan(alpha_index):
                        flag = False

                    markov_shuffles[group][fname][
                        'alpha_index_{}_{}'.format(key, index)
                    ] = alpha_index

            markov_shuffles[group][fname]['flag'] = flag

    return markov_shuffles


def get_essential_data(rna_molecules, markov_shuffles):
    essential_data = {}
    for group in rna_molecules:
        essential_data[group] = {}
        for fname in tqdm(rna_molecules[group]):
            molecule = rna_molecules[group][fname]
            results = {'length': len(molecule.primary_structure)}
            flag = markov_shuffles[group][fname]['flag']
            for key in molecule.secondary_structure.keys():
                for index in ['T-S', 'D-S']:
                    ambiguity_index = molecule.get_ambiguity_index(key, index)
                    if np.isnan(ambiguity_index):
                        flag = False

                    results[
                        'ambiguity_index_{}_{}'.format(key, index)
                    ] = ambiguity_index
                    results['alpha_index_{}_{}'.format(key, index)] = markov_shuffles[
                        group
                    ][fname]['alpha_index_{}_{}'.format(key, index)]

            if flag:
                essential_data[group][fname] = results

    return essential_data
