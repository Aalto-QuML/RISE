# Robust Simulation-based Inference under missing data (RISE)

The code is built on the top of the following repo. Please follow the guidelines in the reference repository to install the requisite packages.

Ref: [Learning Robust Statistics for Simulation-Based Inference Under Model Misspecification](https://github.com/huangdaolang/robust-sbi)

Note: Please change the device (GPU or CPU) in the files accordingly, depending on which you are running.

## Baselines

- We utilized the official code implementation of [Simformer](https://github.com/mackelab/simformer) to run the baseline.
- For NPE-NN (Lueckmann et. al 2017), we followed code implementation  [here](https://github.com/mackelab/delfi).
- For Wang et. al 2024, we utilize train the NPE with the binary mask indicator (as described in [official paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012184)), and can be run by 

```
python -u train_glm_wang/_glu_wang.py --degree degree --type mcar
```


## Running RISE

To run the task on GLU and GLM dataset for mcar/mnar under a certain degree run,

```
python -u train_glm/glu.py --degree degree --type mcar/mnar
```


## License
This code is released under the MIT License.
