# ATON

This is the source code of submission 508.  


### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments

### how to use?
#####1. For ATON, COIN, SHAP, and LIME
1. modify variant `algorithm_name` in `main.py` (support algorithm: `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

#####2. For ATON+, COIN+  
use `--w2s_ratio auto` to run ATON+  
use `--w2s_ratio pn` to run COIN+

#####3. For competitor SiNNE
please run `main_sinne.py` 



### requierments
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
```


### References
- datasets are from ODDS, a outlier detection datasets library (http://odds.cs.stonybrook.edu/), and kaggle platform (https://www.kaggle.com/)
- the source code of competitor COIN is publicly available in github. 