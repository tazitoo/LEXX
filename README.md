# LEXX - Large-scale EXperimentation for XAI
![LEXX spaceship](https://alchetron.com/cdn/lexx-0176a8e0-a0eb-4690-b102-6943540c8c8-resize-750.jpeg)
### Goals:
1. leverage the work from other repos (like [here](https://github.com/kathrinse/TabSurvey), or [here](https://github.com/naszilla/tabzilla)) to facilitate larger experimentation on empirical datasets (if it can settle the debate on what performs best on tabular data, why not try to run XAI + fidelity metrics on is?) 
2. We get for free: multiple datasets, multiple models

   
### What we will need:
1. running an XAI method on each dataset/model along with hyper parameter tuning (using GALE?) to generate a sequence of known / ranked fidelity metrics (e.g. using KernelSHAP with a range of n_sample values)
2. bracketing those attributions with *best* and *worst*
3. running a fidelity metric on each dataset/model/XAI method to recover the rankings
4. summarization over all datasets/models/XAI methods
5. meta-feature extraction (using pymfe)
6. grouping over metafeatures for datasets to determine when a ranking performs well.

### Starting point  
1. using tabzilla as the startin point
2.  Set up the python environment:  
    `conda create -n tabz python==3.10`  
   `conda activate tabz`  
   `python -m pip install -r pip_requirements.txt`
3. Download / preprocess data:  
   `python tabzilla_data_preprocessing.py --process_all`
