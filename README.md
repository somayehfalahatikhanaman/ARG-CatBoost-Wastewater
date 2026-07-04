# CatBoost-ARG-Prediction

## LOGO-Validated CatBoost Framework for Environmental and Omics-Based Prediction of Antibiotic Resistance Genes

This repository accompanies the manuscript:

**LOGO-Validated CatBoost Framework for Environmental and Omics-Based Prediction of Antibiotic Resistance Genes in Urban Wastewater**

## Overview

This repository contains the Python scripts used to reproduce the CatBoost-based machine learning framework developed for predicting antibiotic resistance gene (ARG) occurrence from integrated environmental and metagenomic data.

The workflow includes:

- Metagenomic ARG preprocessing
- NASA POWER environmental data integration
- CatBoost model development
- Leave-One-Group-Out (LOGO) cross-validation
- 5-fold cross-validation
- SHAP explainability analysis

---

## Repository Structure

scripts/
    01_catboost_preprocessing.py
    02_catboost_logo_validation.py
    03_catboost_kfold_validation.py
    04_catboost_shap_analysis.py
    05_nasa_power_api.py

data/
    README.md

results/
    README.md

---

## Data Availability

Raw metagenomic sequencing reads are publicly available through the European Nucleotide Archive (ENA).

BioProject:

**PRJEB68319**

Processed feature matrices and supplementary datasets are available from:

**Mendeley Data**

https://doi.org/10.17632/7b929fysdb.1

---

## Requirements

Python 3.11

Required packages are listed in **requirements.txt**.

---

## Citation

If you use this repository, please cite the associated publication.
