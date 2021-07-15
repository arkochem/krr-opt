#### Before the installation, create and activate your conda env with python 3.9:
```console
conda create -n ENV_NAME python=3.9
conda activate ENV_NAME

conda install -c conda-forge 'joblib>=1.3.0' 'scipy>=1.11.0' 'numpy>1.23.0,<1.24.0' 'matplotlib>=3.7.0' 'scikit-learn>=1.5.0'
```
or use any existing environment with **python 3.9**.

#### You will need the fixed fork of SHGO optimizer (v0.5.1):
```console
pip install -U git+https://github.com/arkochem/shgo.git@v0.5.1#egg=shgo
```

#### To install the krr-opt package: 
Run the following (do not forget to activate your environment):
```console
pip install -U git+https://github.com/arkochem/krr-opt.git@QUED#egg=krr_opt
```

### TODO

- [ ] Tests, including sklearn.kernel_ridge.KernelRidge and numerical kernel values along with their derivatives 
- [ ] Update docstrings, especially for methods where derivations are not obvious. Comment on matrix/vector dims
- [ ] Update SHGO (https://github.com/Stefan-Endres/shgo) to the latest version (or maybe use one from SciPy)
- [ ] Optimize sigma/alpha params in a log scale and add possibility to set custom optimization bounds
- [ ] Create method fit_optimize() with args X_val and y_val initially set to None (only fit by default)
