# ATON

This is the source code of a submission of WWW'21 (Submission Id: 508)  
The source code of the proposed method ATON in included in folder `model_aton`.  
Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearch for this new practical probelm.*



### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments

### How to use?
##### 1. For ATON, and competitor COIN, SHAP, and LIME
1. modify variant `algorithm_name` in `main.py` (support algorithm: `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For ATON+ and competitor COIN+ 
1. modify variant `algorithm_name` in `main.py` to `aton` or `coin`  
2. use `python main.py --path data/ --w2s_ratio auto --runs 10` to run ATON+  
   use `python main.py --path data/ --w2s_ratio pn --runs 10` to run COIN+  

##### 3. For competitor SiNNE
please run `main_sinne.py` 

### args of main.py
- `--path [str]`        - the path of data folder or an individual data file (in csv format)  
- `--gpu  [True/False]` - use GPU or not
- `--runs [int]`         - how many times to run a method on each dataset (we run 10 times and report average performance in our submission)
- `--w2s_ratio [auto/real_len/pn]`  - how to transfer feature weight to feature subspace
- `--eval [True/False]` - evaluate or not, use False for scalability test


### Requierments
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
```


### References
- datasets are from ODDS, a outlier detection datasets library (http://odds.cs.stonybrook.edu/), and kaggle platform (https://www.kaggle.com/)
- the source code of competitor COIN is publicly available in github. 
