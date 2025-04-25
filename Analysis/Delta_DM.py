import tensorflow.compat.v1 as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import os


#Random seed
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.set_random_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(2013)

######################### Data Preprocessing #########################
print("`````````` Data Preprocessing ``````````")
#Load Model
model = keras.models.load_model('/demo/Delta_DM/DM_2013/')
#Load whole genome SNP
#annotation
snp_cpg_anno_raw = pd.read_csv('/demo/Delta_DM/snp_cpg_pairs.csv',index_col=0)
#sequence
snp_refseq = np.load('/demo/Delta_DM/snp_refseq.npy')
snp_varseq = np.load('/demo/Delta_DM/snp_varseq.npy')

#预处理annotation
snp_cpg_anno = snp_cpg_anno_raw.iloc[:,7:-6]
#preprocessing
#Normolization(list 15 16 18 19 20)
snp_cpg_anno.iloc[:,15:17]=snp_cpg_anno.iloc[:,15:17].apply(
    lambda x:(x-x.mean())/(x.std()))
snp_cpg_anno.iloc[:,18:21]=snp_cpg_anno.iloc[:,18:21].apply(
    lambda x:(x-x.mean())/(x.std()))
#delete Neighbour features
a = snp_cpg_anno.iloc[:,0:14]
b = snp_cpg_anno.iloc[:,20:]
snp_cpg_anno = pd.concat([a,b],axis=1)
snp_cpg_anno = snp_cpg_anno.to_numpy()

#处理sequence
snp_refseq = np.reshape(snp_refseq,[-1,414,4,1])
snp_varseq = np.reshape(snp_varseq,[-1,414,4,1])
print("`````````` Finish ``````````")
######################### Finish #########################

######################### DM Model Evaluating #########################
print("`````````` Model Evaluating ``````````")
ref_pred = model.predict([snp_refseq,snp_cpg_anno])
ref_prob = ref_pred[:, 1]
var_pred = model.predict([snp_varseq,snp_cpg_anno])
var_prob = var_pred[:, 1]

SNP_effect = np.abs(var_prob-ref_prob)
np.save('./DeltaDM_effectsize',SNP_effect)
print("`````````` Finish ``````````")
######################### Finish #########################

