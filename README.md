# ATON

This is the source code of a submission of WWW'21 (Submission Id: 508)  



### update news

**[Nov 28] Following the reviewer's suggestion, we add two new contenders: Integrated Gradients [1] and Anchor [2].**

[1] Axiomatic attribution for deep networks. In ICML. 2017.

[2] Anchors: High Precision Model-Agnostic Explanations. In AAAI. 2018.

We use the implementations of these two methods in package `alibi`. 

Please see two classes in the folder `model_iml` (short for interpretable machine learning). 



### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments

### How to use?
##### 1. For ATON and competitor COIN, SHAP, and LIME
1. modify variant `algorithm_name` in `main.py` (support algorithm: `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For ATON' and competitor COIN' 
1. modify variant `algorithm_name` in `main.py` to `aton` or `coin`  
2. use `python main.py --path data/ --w2s_ratio auto --runs 10` to run ATON'  
   use `python main.py --path data/ --w2s_ratio pn --runs 10` to run COIN'  

##### 3. For competitor SiNNE
please run `main_sinne.py` 

### args of main.py
- `--path [str]`        - the path of data folder or an individual data file (in csv format)  
- `--gpu  [True/False]` - use GPU or not
- `--runs [int]`         - how many times to run a method on each dataset (we run 10 times and report average performance in our submission)
- `--w2s_ratio [auto/real_len/pn]`  - how to transfer feature weight to feature subspace
- `--eval [True/False]` - evaluate or not, use False for scalability test  
... (other hypter-parameters of different methods. You may want to use -h to check the corresponding hypter-parameters after modifing the `algorithm_name`)  

### Requirements
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```


### Ground-truth annotations
Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearchs on this new practical probelm.*  

You may find that each dataset has three annotation files, please refer to the detailed annotation generation process in our submission. We detailedly introduced it in Section 5.1.4:  

**How to generate the ground-truth annotations:**
>  We employ three different kinds of representative outlier detection methods (i.e., ensemble-based method iForest, probability-based method COPOD, and distance-based method HBOS) to evaluate outlying degree of real outliers given every possible subspace. A good explanation for an outlier should be a high-contrast subspace that the outlier explicitly demonstrates its outlierness, and outlier detectors can easily and certainly predict it as an outlier in this subspace. Therefore, the ground-truth interpretation for each outlier is defined as the subspace that the outlier obtains the highest outlier score among all the possible subspaces.



### References
- datasets are from ODDS, a outlier detection datasets library (http://odds.cs.stonybrook.edu/), and kaggle platform (https://www.kaggle.com/)
- the source code of competitor COIN is publicly available in github. 
