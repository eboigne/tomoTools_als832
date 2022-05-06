# TomoTools

A set of Python scripts used to post-process CT datasets acquired at the 832 beamline at the Advanced Light Source (ALS).

## Install

Install using conda, create and activate an environment:

`conda create -n tomoTools`

`conda activate tomoTools`

Use conda to install mamba:

`conda install -c conda-forge mamba python=3.6`

Use mamba to install dependencies (faster than conda, but still takes some time):

`mamba install -c conda-forge -c simpleitk -c anaconda tifffile numpy scipy matplotlib jupyter numexpr astropy h5py simpleitk scikit-image jedi=0.17`

Install PyTorch, make sure that a GPU version is installed (check [the official website](https://pytorch.org/) for more instructions):

`mamba install -c pytorch pytorch cudatoolkit=10.1`

`mamba install -c astra-toolbox/label/dev astra-toolbox`

Then clone the scripts, and run them in a local notebook:

`git clone https://github.com/eboigne/tomoTools_als832.git`

`jupyter-notebook`

## Debug
If kernel dies every time the Astra algorithm is run, try uninstalling and re-installing Astra:

`mamba remove astra-toolbox`

`mamba install -c astra-toolbox/label/dev astra-toolbox`

