#Select features by comparining two groups
import os,csv,re, time
import pickle
import random
import pandas as pd
import numpy as np
from scipy import stats
import scanpy as sc
from scipy.sparse import issparse

def rank_genes_groups(input_adata, target,  label_col, non_target="rest", two_sides=False, logged=True):
	adata=input_adata.copy()
	if non_target=="rest":
		adata.obs["target"]=((adata.obs[label_col]==target)*1).astype('category')
		print("Data contains: ", set(adata.obs[label_col]))
	else:
		adata=adata[adata.obs[label_col].isin(non_target+[target])]
		print("Data contains: ", set(adata.obs[label_col]))
		adata.obs["target"]=((adata.obs[label_col]==target)*1).astype('category')
	sc.tl.rank_genes_groups(adata, groupby="target",reference="rest", n_genes=adata.shape[1],method='wilcoxon')
	#rank genes for characterizing groups. rank genes based on what scores?
	#--------------------------Target--------------------------
	#pvals_adj: corrected p-values
	pvals_adj=[i[1] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
	genes=[i[1] for i in adata.uns['rank_genes_groups']["names"]]
	if issparse(adata.X):
		obs_tidy=pd.DataFrame(adata.X.A)
	else:
		obs_tidy=pd.DataFrame(adata.X)
	obs_tidy.index=adata.obs["target"].tolist()
	obs_tidy.columns=adata.var.index.tolist()
	obs_tidy=obs_tidy.loc[:,genes]
	# 1. compute mean value
	mean_obs = obs_tidy.groupby(level=0).mean() #here is log mean
	# 2. compute fraction of cells having value >0
	obs_bool = obs_tidy.astype(bool) #python boolean only has two possible values: True or False
	fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
	# 3. compute fold change.
	if logged: #The adata already logged
		fold_change=np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
	else:
		fold_change = (mean_obs.loc[1] / (mean_obs.loc[0]+ 1e-9)).values #why need to add a very small value
	df1 = {'genes':genes,'in_group_fraction':fraction_obs.loc[1].tolist(),"out_group_fraction":fraction_obs.loc[0].tolist(),"in_out_group_ratio":(fraction_obs.loc[1]/fraction_obs.loc[0]).tolist(),"in_group_mean_exp":mean_obs.loc[1].tolist(),"out_group_mean_exp":mean_obs.loc[0].tolist(),"fold_change":fold_change.tolist(), "pvals_adj":pvals_adj}
	df1 = pd.DataFrame(data=df1)
	if two_sides==False:
		return df1
	else:
		#--------------------------Rest--------------------------
		pvals_adj=[i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
		genes=[i[0] for i in adata.uns['rank_genes_groups']["names"]]
		if issparse(adata.X):
			obs_tidy=pd.DataFrame(adata.X.A)
		else:
			obs_tidy=pd.DataFrame(adata.X)
		obs_tidy.index=((adata.obs["target"]==0)*1).tolist()
		obs_tidy.columns=adata.var.index.tolist()
		obs_tidy=obs_tidy.loc[:,genes]
		# 1. compute mean value
		mean_obs = obs_tidy.groupby(level=0).mean()
		# 2. compute fraction of cells having value >0
		obs_bool = obs_tidy.astype(bool)
		fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
		# compute fold change.
		if logged: #The adata already logged
			fold_change=np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
		else:
			fold_change = (mean_obs.loc[1] / (mean_obs.loc[0]+ 1e-9)).values
		df0 = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(), "out_group_fraction":fraction_obs.loc[0].tolist(),"in_out_group_ratio":(fraction_obs.loc[1]/fraction_obs.loc[0]).tolist(),"in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),"fold_change":fold_change.tolist(), "pvals_adj":pvals_adj}
		df0 = pd.DataFrame(data=df0)
		return df1, df0





