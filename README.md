# Forward-Selection-With-Machine-Learning-Algorithm (Wrapper Method)
Implementation of Feature Forward Selection with RF 

# Dependency
pip install mlxtend

# Input File
CSV file Format: 
1. row-instance(observation vector)  column-attribute(feature vector)

# Using Wrapper Method ---- Forward Selection to do the feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


sfs = SFS(base_model,      
         k_features = 3,  
          forward= True,   
          floating = False,
          verbose= 2,
          scoring= 'accuracy',
          cv = 4,          
          n_jobs= -1      
         ).fit(X_train, y_train)
 
base_model: sklearn classifer or regressor

k_features: feature number of feature subset

Forward: True-Forward Selection False-Backward Selection

cv: cross_validation

n_jobs: The number of CPUs to use for evaluating different feature subsets in parallel. e.g "-1" means all CPUs

# Output:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done   4 out of   4 | elapsed:    4.5s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done   4 out of   4 | elapsed:    4.5s finished

[2019-12-03 07:39:39] Features: 1/3 -- score: 0.9496941045606229[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    4.4s finished

[2019-12-03 07:39:43] Features: 2/3 -- score: 0.9255005561735261[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   2 | elapsed:    3.0s finished

[2019-12-03 07:39:47] Features: 3/3 -- score: 0.9413422321097515

sfs.k_score_ = 0.9413422321097515  ##best score in all subset

# reference
mlxtend: Documentation of official website
http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#sequential-forward-selection-sfs
