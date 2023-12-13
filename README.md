# LEXX - <ins>L</ins>arge-scale <ins>EX</ins>perimentation for <ins>X</ins>AI
![LEXX spaceship](https://alchetron.com/cdn/lexx-0176a8e0-a0eb-4690-b102-6943540c8c8-resize-750.jpeg)
### Goals:
1. leverage the work from other repos (like [here](https://github.com/kathrinse/TabSurvey), or [here](https://github.com/naszilla/tabzilla)) to facilitate larger experimentation on empirical datasets (if it can settle the debate on what performs best on tabular data, why not try to run XAI + fidelity metrics on is?) 
2. We get for free: multiple datasets, multiple models
3. We should focus on the 39 hard datasets - where a strong baseline (linear) model does not perform well.
4. These experiments on empirical data are intended to be complementary to the synthetic dataset experiments  - aka 'this is how it works on your dataset'.


### Getting started  
Using [my fork](https://github.com/tazitoo/tabzilla) of tabzilla as the starting point.  Use the `dev` branch, it has some capability to save the models from the HP tuning. 

1.  Set up the python environment:  
    ```
    conda create -n tabz python==3.10 -y  
    conda activate tabz 
    python -m pip install -r pip_requirements.txt
    python -m ipykernel install --user --name=tabz    
    ```
2. Download / preprocess data:  
   ```
   python tabzilla_data_preprocessing.py --process_all
   ```
3. create a model (needs experiment config, model type, and dataset directory)  
   ```
   python tabzilla_experiment.py --experiment_config tabzilla_experiment_config.yml --model_name XGBoost --dataset_dir datasets/openml__acute-inflammations__10089
   ```

### What we will need:
1. running an XAI method on each dataset/model along with hyper parameter tuning (using GALE?) to generate a sequence of known / ranked fidelity metrics (e.g. using KernelSHAP with a range of n_sample values)
2. bracketing those attributions with *best* and *worst*
3. running a fidelity metric on each dataset/model/XAI method to recover the rankings
4. summarization over all datasets/models/XAI methods
5. meta-feature extraction (using pymfe)
6. grouping over metafeatures for datasets to determine when a ranking performs well.
   
### TODO on workflow  
Get end to end process working for 1 dataset, 1 model type (e.g. XGBoost) and 1 XAI method.  This involves:  
- [x] download dataset (see instructions above)  
- [x] build model (see above)
- [ ] save model (some work - will save to `Tabzilla/output/<model_type>/<dataset name>/`  
- [ ] or save optuna HPs?  (then we can do train/test split again, save them, and refit)  
- [ ] use model + test split + XAI method + HP tuning to generate known ranked local attributions  


### Notes  
-- if you want to run a NAM, you need to clone the [nam repo](https://github.com/AmrMKayid/nam) and pip install it.  Requires GPU(?) Need to add this to `pip-requirements.txt` ...
