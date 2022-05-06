# TomoTools

A set of Python scripts used to post-process CT datasets acquired at the 832 beamline at the Advanced Light Source (ALS).

## Install

Install using conda, create and activate an environment:
`conda create -n tomoTools`
`conda activate tomoTools`
Use conda to install mamba:
`conda install -c conda-forge mamba python=3.6`
Use mamba to install dependencies (faster than conda):
`mamba install -c conda-forge -c pytorch -c simpleitk -c anaconda  tifffile numpy scipy matplotlib jupyter numexpr astropy h5py simpleitk scikit-image tomopy jedi=0.17 pytorch cudatoolkit=10.1`
Then:
`mamba install -c astra-toolbox/label/dev astra-toolbox`
`git clone https://github.com/eboigne/tomoTools_als832.git`
`jupyter-notebook`
And then you can open the notebooks in the cloned tomoTools folder.

## Debug
If kernel dies every time the Astra algorithm is run, try uninstalling and re-installing astra:
`mamba remove astra-toolbox`
`mamba install -c astra-toolbox/label/dev astra-toolbox`

Error with skimage (needed by Tomopy at least), using numpy features from >= 1.16, but above installs numpy 1.12. Thus forcing the installation:
`mamba install numpy=1.16`

