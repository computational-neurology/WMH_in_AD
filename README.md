# Code for: *Leone R., Kobeleva X., White Matter Hyperintensities Contribute to Early Cortical Thinning in Addition to Tau in Aging, Neurobiology of Aging, 2025*

This repository contains the code used to generate the figures and tables presented in the manuscript *"White Matter Hyperintensities Contribute to Early Cortical Thinning in Addition to Tau in Aging"*, published in *Neurobiology of Aging* and available [here](https://www.sciencedirect.com/science/article/abs/pii/S0197458025001216).

## Important Note

Data used in the preparation of this article was obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu).
Due to data sharing restrictions, the original dataset used in this study cannot be publicly released. The code is provided to support transparency of the analysis workflow.

## Repository Structure

- **`main_analyses.ipynb`**  
  Jupyter notebook containing the main analysis pipeline, including all steps used to generate the figures and tables in the manuscript.

- **`utils.py`**  
  A collection of utility functions used in the main notebook for data handling, statistical analysis, and visualization.

- **`plot_disconnectivity.py`**  
  A standalone script for plotting the disconnectivity diagram. This script is kept separate due to compatibility issues with some packages used in the Jupyter environment.

- **`run_LQT.R`** 
  R code to run the lesion quantification toolkit software in R. R code of the LQT is available [here](https://github.com/jdwor/LQT). Note that the Lesion Quantification Toolkit pipeline was run in R Studio using Windows 11!

