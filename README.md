# SMSM Supplementary Repository
Supplementary materials for the paper "Hierarchical Semi-Markov Smooth Models of Latent Neural States"
by Krause, van Rij, and Borst (submitted).

## Performing the analysis:

To run the analysis first set up a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
with python > 3.10 and install the following Python packages (versions are important):
 - mssm==1.2.0
 - mssmViz==0.1.48
 - mne==1.6.0
 - hmp==0.4.1
 - arviz==0.22.0
 - dotenv==0.9.9
 - multiprocess==0.70.18

**Note:** ``mssm`` version 1.2.0 has not yet been made available on PyPi but must instead be installed
from Github. See [here](https://github.com/JoKra1/mssm/tree/1.2.0) for instructions. The specific version of ``hmp``
might also have to be installed from GitHub. See [here](https://github.com/GWeindel/hmp).

Then you can:
 - Clone this repository
 - Get the RT data ("rt_dat.csv") and EEG data ("data4HMM_baseline_LD_excl_6_15_bs_100_.mat") from Krause et al. (2024)


You will then need to set up an environment file at the top level of this repository called ``.env``
with variables defining where to store results, where to find the data from Krause et al. (2024), and
how many CPU cores to use during the analysis. For example, my ``.env`` file looks like this:

```
data_path="/Users/joshmac/Documents/repos/HSMM_LD_EEG/data/"
results_path="/Users/joshmac/Documents/repos/SMSM_supplementary/models/"
n_cores=4
overwrite=0
```

**Note** to maximize performance, ``n_cores`` should be set to half the number of physical cores
available on your system.
 
Then, all you need to do is run ``SMSM_eeg_analysis.ipynb``.