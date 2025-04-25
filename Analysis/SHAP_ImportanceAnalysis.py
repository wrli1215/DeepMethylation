import tensorflow.compat.v1 as tf
from tensorflow import keras
import random
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

#Random seed
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.set_random_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(2023)

######################### Data Preprocessing #########################
print("`````````` Data Preprocessing ``````````")
# Load Model
model = keras.models.load_model('/demo/importance_analysis/DM_epi_model/')
# Load annotation （分组织）
annotation_train = pd.read_csv('/demo/DM_train_evaluation/annotation_train.csv', index_col=0)
annotation_test = pd.read_csv('/demo/DM_train_evaluation/annotation_test.csv', index_col=0)
annotation_train = annotation_train.iloc[:, 4:]
annotation_test = annotation_test.iloc[:, 4:]

# preprocessing
# Normolization(list 15 16 18 19 20)
annotation_train.iloc[:, 15:21] = annotation_train.iloc[:, 15:21].apply(
    lambda x: (x - x.mean()) / (x.std()))
annotation_test.iloc[:, 15:21] = annotation_test.iloc[:, 15:21].apply(
    lambda x: (x - x.mean()) / (x.std()))
# delete Neighbour features
annotation_train = pd.concat([annotation_train.iloc[:, 0:14], annotation_train.iloc[:, 20:]], axis=1)
annotation_test = pd.concat([annotation_test.iloc[:, 0:14], annotation_test.iloc[:, 20:]], axis=1)

annotation_train = annotation_train.to_numpy()
annotation_test = annotation_test.to_numpy()

# 打乱训练集
index = [i for i in range(len(annotation_train))]
np.random.shuffle(index)
annotation_train = annotation_train.iloc[index,:]


print("`````````` Finish ``````````")
######################### Finish #########################


######################### Importance Analysis #########################
print("`````````` Importance Analysis ``````````")
#use SHAP to explain
shap.initjs()
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the "shap_ADDV2" problem but another one will appear
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer

# select a set of background examples to take an expectation over
background = annotation_train
explainer = shap.DeepExplainer(model,background)
shap_values = explainer.shap_values(annotation_test)

feature_names=['CpG_island','CpG_shore','CpG_shelf','CpG_opensea','promoter','1to5kb','5UTR','3UTR','1stexon','exon','intron','enhancer','GeneBody','IntergenicRegion',
                  'GC_content','CTCF','POLR2A','H3K4me1','H3K4me3','H3K9me3','H3K27ac','H3K27me3','H3K36me3','DNase-seq','ATAC-seq']
#The SHAP(importance) summary plot
fig = shap.summary_plot(shap_values[1], annotation_test,show=False,max_display=25,alpha=0.1,sort=False,feature_names=feature_names)
plt.savefig('./DMepi_shap_summary_plot.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.clf()
#The top 10 features mean|SHAP value(importance)| ranking plot
shap.summary_plot(shap_values[1], annotation_test, plot_type="bar",show=False,max_display=10,feature_names=feature_names)
plt.savefig('./DMepi_shap_mean_plot.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.clf()
print("`````````` Finish ``````````")
######################### Finish #########################
