# Introduction

This repository contains the code to reproduce the experimental results in the paper ["Base-pair Ambiguity and the Kinetics of RNA Folding"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3303-6).

# Setting up the environment

1. Clone this repository
```
git clone https://github.com/StannisZhou/rna_statistics.git
```

2. Set up the virtual environment
```
cd rna_statistics
conda env create -f environment.yml
source activate rna_statistics
```

# Reproduce the results

Use
```
streamlit run main.py
```
to reproduce all the results using cached results in the `intermediate` folder. This would open up a browser session and interactively generate all the supplementary results.

A pre-generated report containing all the results is also available at `report/index.html`.

# Regenerate all the results from scratch

To facilitate the exploration of the results, we included cached results in the `intermediate` folder. To regenerate all the results from scratch, a few additional external dependicies are needed:

1. Use the [standalone implementation](http://www.ibi.vu.nl/programs/k2nwww/static/k2n_standalone.tgz) of [Knotted to Nested](http://www.ibi.vu.nl/programs/k2nwww/) to remove pseudoknots from the comparative secondary structures available under `data`. A copy of the pseudoknot-free comparative secondary structures (generated using the IR option) is available under `data_without_pseudoknots`. Note that you need to make simple modifications to the bpseq parser in the standalone implementation of Knotted to Nested so that it works with the data available under `data`, and use the code under python 2.7.
2. Set up the [RNA Secondary Structure Analyser](http://www.rnasoft.ca/strand/download/RNAAnalyser.tar.gz), available at the [RNA STRAND database website](http://www.rnasoft.ca/strand/). You might need to change line 42 of `interval.cpp`, and properly set X11 related options in `Makefile` (e.g. properly set `X11_INCLUDE` and change X11 related options in `LFLAGS`) in order to successfully compile the analyser. The code assumes the analyser is available in the RNAAnalyser folder under you home directory. If it's placed in a different place, please change line 14 of `data.py` accordingly.
3. Install the [Vienna RNA package](https://www.tbi.univie.ac.at/RNA/). Make sure `RNAfold`  and `RNAPKplex` are available at the commandline.

After setting up these external dependicies and generate the `data_without_pseudoknots` folder, you can remove the `intermediate` folder and use
```
python run_experiments.py
```
to regenerate the cached results in `intermediate`.
