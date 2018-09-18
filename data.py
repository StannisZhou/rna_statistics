import os
import re
import sys
import numpy as np
import rna
from tqdm import tqdm


ALPHABET = ['A', 'G', 'C', 'U']


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
                    header, primary_structure, secondary_structure = parse_bpseq(file_name)
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
                    rna_molecules[group][name]['secondary_structure'] = secondary_structure

    return rna_molecules


def get_rna_molecules(rna_raw_data, K, r):
    rna_molecules = {}
    for group in rna_raw_data:
        rna_molecules[group] = {}
        for fname in tqdm(rna_raw_data[group]):
            molecule = rna.RNA(
                group, rna_raw_data[group][fname]['primary_structure'],
                rna_raw_data[group][fname]['secondary_structure'], 'wobble',
                K, r
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
            molecule = rna_molecules[group][fname]
            shuffles = rna.get_markov_shuffles(
                molecule.primary_structure, molecule.case, molecule.K, molecule.r, parallel
            )
            markov_shuffles[group][fname] = shuffles

    return markov_shuffles


def get_essential_data(rna_molecules, markov_shuffles):
    essential_data = {}
    for group in rna_molecules:
        essential_data[group] = {}
        for fname in tqdm(rna_molecules[group]):
            molecule = rna_molecules[group][fname]
            if markov_shuffles:
                shuffles = markov_shuffles[group][fname]

            results = {
                'length': len(molecule.primary_structure)
            }
            flag = True
            for key in ['comparative', 'mfe']:
                for index in ['T-S', 'D-S']:
                    ambiguity_index = molecule.get_ambiguity_index(key, index)
                    if np.isnan(ambiguity_index):
                        flag = False

                    results[
                        'ambiguity_index_{}_{}'.format(key, index)
                    ] = ambiguity_index
                    if markov_shuffles:
                        alpha_index = molecule.get_alpha_index(key, index, shuffles)
                        if np.isnan(alpha_index):
                            flag = False

                        results[
                            'alpha_index_{}_{}'.format(key, index)
                        ] = alpha_index

            if flag:
                essential_data[group][fname] = results

    return essential_data
