import os
import subprocess
import numpy as np
import itertools
from collections import Counter
import ushuffle
import multiprocess as mp


RNA_GROUPS = [
    'Group_I_Introns', 'Group_II_Introns', 'tmRNAs', 'SRP_RNAs',
    'RNase_Ps', '16s_ribosomal', '23s_ribosomal'
]
SHORTEST_LOOP = 3


class RNA(object):
    def __init__(self, group, primary_structure, secondary_structure_comparative, case, K, r):
        assert group in RNA_GROUPS
        assert case in ['wc_only', 'wobble']
        self.group = group
        self.primary_structure = primary_structure
        self.secondary_structure = {
            'comparative': secondary_structure_comparative,
            'mfe': vienna_rnafold(primary_structure)
        }
        self.case = case
        self.K = K
        self.r = r

    def evaluate_local_ambiguities(self):
        self.local_ambiguities = get_local_ambiguities(self.primary_structure, self.r, self.case)

    def label_locations_using_secondary_structures(self):
        r = self.r
        paired_indicator = {}
        n_paired_nucleotides = {}
        masks = {
            'single': {},
            'double': {},
            'transitional': {}
        }
        N = len(self.primary_structure)
        for key in ['comparative', 'mfe']:
            paired_indicator[key] = (self.secondary_structure[key] > -1).astype(int)
            n_paired_nucleotides[key] = np.zeros(N - r + 1)
            for ii in range(N - r + 1):
                n_paired_nucleotides[key][ii] = np.sum(paired_indicator[key][ii:ii + r])

            masks['single'][key] = (n_paired_nucleotides[key] == 0)
            masks['double'][key] = (n_paired_nucleotides[key] == r)
            masks['transitional'][key] = (n_paired_nucleotides[key] < r) * (n_paired_nucleotides[key] > 0)
            total_locations = 0
            for region in masks:
                total_locations += np.sum(masks[region][key])

            assert total_locations == N - r + 1

        self.masks = masks

    def get_mean_ambiguity(self, key, region, local_ambiguities_list, exclude_zero=False):
        assert key in ['comparative', 'mfe']
        assert region in ['single', 'double', 'transitional']
        assert hasattr(self, 'masks')
        assert local_ambiguities_list.ndim == 2
        n_local_ambiguities = local_ambiguities_list.shape[0]
        mask = np.tile(
            self.masks[region][key].reshape((1, -1)), (n_local_ambiguities, 1)
        )
        if exclude_zero:
            mask = (local_ambiguities_list > 0) * mask

        non_empty_mask_ind = (np.sum(mask, axis=1) > 0)
        mean_ambiguities = np.zeros(n_local_ambiguities)
        mean_ambiguities[non_empty_mask_ind] = np.average(
            local_ambiguities_list[non_empty_mask_ind], axis=1, weights=mask[non_empty_mask_ind]
        )
        mean_ambiguities[~non_empty_mask_ind] = np.nan
        return mean_ambiguities

    def get_ambiguity_index(self, key, index):
        assert key in ['comparative', 'mfe']
        assert index in ['T-S', 'D-S']
        local_ambiguities_list = self.local_ambiguities.reshape((1, -1))
        if index == 'T-S':
            ambiguity_index = self.get_mean_ambiguity(
                key, 'transitional', local_ambiguities_list
            ) - self.get_mean_ambiguity(
                key, 'single', local_ambiguities_list
            )
        else:
            ambiguity_index = self.get_mean_ambiguity(
                key, 'double', local_ambiguities_list, True
            ) - self.get_mean_ambiguity(
                key, 'single', local_ambiguities_list, True
            )

        return ambiguity_index[0]

    def get_alpha_index(self, key, index, shuffles=None):
        assert key in ['comparative', 'mfe']
        assert index in ['T-S', 'D-S']
        ambiguity_index = self.get_ambiguity_index(key, index)
        if shuffles:
            _, shuffle_ambiguities = shuffles
        else:
            _, shuffle_ambiguities = get_markov_shuffles(
                self.primary_structure, self.case, self.K, self.r
            )

        if index == 'T-S':
            shuffle_ambiguity_index = self.get_mean_ambiguity(
                key, 'transitional', shuffle_ambiguities
            ) - self.get_mean_ambiguity(
                key, 'single', shuffle_ambiguities
            )
        else:
            shuffle_ambiguity_index = self.get_mean_ambiguity(
                key, 'double', shuffle_ambiguities, True
            ) - self.get_mean_ambiguity(
                key, 'single', shuffle_ambiguities, True
            )

        alpha = (1 + np.sum(shuffle_ambiguity_index <= ambiguity_index)) / (1 + self.K)
        return alpha


def get_local_ambiguities(primary_structure, r, case):
    PAIRING_PARTERS = {
        'wc_only': {'A': ['U'], 'G': ['C'], 'C': ['G'], 'U': ['A']},
        'wobble': {'A': ['U'], 'G': ['C', 'U'], 'C': ['G'], 'U': ['A', 'G']}
    }[case]
    N = len(primary_structure)
    segments_list = [
        ''.join(primary_structure[ii:ii + r]) for ii in range(N - r + 1)
    ]
    segments_dict = dict(Counter(segments_list))
    local_ambiguities = np.zeros(N - r + 1, dtype=int)
    for ii, segment in enumerate(segments_list):
        lower_end = max(0, ii - r - SHORTEST_LOOP)
        upper_end = min(N - r + 1, ii + r + SHORTEST_LOOP)
        not_viable_dict = dict(Counter(segments_list[lower_end:upper_end]))
        complementary_nucleotides = [
            PAIRING_PARTERS[nucleotide] for nucleotide in reversed(list(segment))
        ]
        list_of_segment_complementary = itertools.product(*complementary_nucleotides)
        list_of_segment_complementary = [
            ''.join(segment_complementary) for segment_complementary in list_of_segment_complementary
        ]
        local_ambiguities[ii] = 0
        for segment_complementary in list_of_segment_complementary:
            local_ambiguities[ii] += segments_dict.get(segment_complementary, 0) -\
                not_viable_dict.get(segment_complementary, 0)

    return local_ambiguities


def vienna_rnafold(primary_sequence):
    """vienna_rnafold

    Wrapper function for the RNAfold program in the Vienna RNA package. Use the
    default settings to fold the rna sequence.

    Parameters
    ----------

    primary_sequence : list
        primary_sequence is a list of all nucleotides.

    Returns
    -------

    secondary_structure : np array
        -1 if unpaired, otherwise nonnegative and indicates the index of the
        pairing partner

    """
    primary_sequence = ''.join(primary_sequence)
    rna_fold = subprocess.Popen(['RNAfold', '--noPS'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    rna_fold_output = rna_fold.communicate(primary_sequence.encode('utf-8'))[0]
    secondary_structure_dot_bracket = rna_fold_output[
        len(primary_sequence) + 1:2 * len(primary_sequence) + 1]
    secondary_structure_dot_bracket = secondary_structure_dot_bracket.decode('utf-8')
    secondary_structure = _parse_dot_bracket(secondary_structure_dot_bracket)
    return secondary_structure


def _parse_dot_bracket(secondary_structure_dot_bracket):
    """_parse_dot_bracket

    Parse the output from the RNAfold program in the Viennna RNA package, which
    is in the dot bracket form

    Parameters
    ----------

    secondary_structure_dot_bracket : string
        secondary_structure_dot_bracket is the secondary structure in the dot
        bracket form

    Returns
    -------

    secondary_structure : np array
        -1 if unpaired, otherwise nonnegative and indicates the index of the
        pairing partner

    """
    secondary_structure = -np.ones(len(secondary_structure_dot_bracket))
    left_bracket = []
    for ii, char in enumerate(secondary_structure_dot_bracket):
        if char == '(':
            left_bracket.append(ii)

        if char == ')':
            partner_location = left_bracket.pop()
            secondary_structure[ii] = partner_location
            secondary_structure[partner_location] = ii

    return secondary_structure


def get_markov_shuffles(primary_structure, case, K, r, parallel=True):
    primary_structure = ''.join(primary_structure).encode('utf-8')
    shuffler = ushuffle.Shuffler(primary_structure, r)
    shuffle_sequences = []
    for _ in range(K):
        shuffle_sequences.append(shuffler.shuffle().decode('utf-8'))

    def _parallel_complementary(shuffle_sequence):
        shuffle_ambiguity = get_local_ambiguities(shuffle_sequence, r, case)
        return shuffle_ambiguity

    N = len(primary_structure)
    shuffle_ambiguities = np.zeros((K, N - r + 1), dtype=int)
    if parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        for ii, shuffle_ambiguity in enumerate(pool.imap(_parallel_complementary, shuffle_sequences)):
            shuffle_ambiguities[ii] = shuffle_ambiguity

        pool.close()
        pool.join()
    else:
        for ii, shuffle_sequence in enumerate(shuffle_sequences):
            shuffle_ambiguity = _parallel_complementary(shuffle_sequence)
            shuffle_ambiguities[ii] = shuffle_ambiguity

    shuffle_sequences = np.array(shuffle_sequences)
    return shuffle_sequences, shuffle_ambiguities
