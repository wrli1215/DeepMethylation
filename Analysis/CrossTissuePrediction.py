import tensorflow.compat.v1 as tf
from tensorflow import keras
import random
import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics


# Evaluate performance of model
def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test, y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc = metrics.average_precision_score(y_test, y_prob)
    auprc_curve = metrics.precision_recall_curve(y_test, y_prob)
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # MCC
    mcc = metrics.matthews_corrcoef(y_test, y_pred)

    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    class_report = metrics.classification_report(y_test, y_pred, target_names=["hypomethylation", "hypermethylation"])
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    model_perf_show = {"auroc": auroc,
                       "auprc": auprc,
                       "accuracy": accuracy, "mcc": mcc,
                       "recall": recall, "precision": precision, "f1": f1,
                       }
    print(model_perf_show)
    print(confusion_matrix)
    model_perf = {"auroc": auroc, "auroc_curve": auroc_curve,
                  "auprc": auprc, "auprc_curve": auprc_curve,
                  "accuracy": accuracy, "mcc": mcc,
                  "recall": recall, "precision": precision, "f1": f1,
                  "class_report": class_report,
                  "confusion_matrix": confusion_matrix,
                  "specificity": (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))}

    return model_perf


# Output result of evaluation
def eval_output(model_perf, output_file):
    with open(output_file, 'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tMCC=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
                (model_perf["auroc"], model_perf["auprc"], model_perf["accuracy"], model_perf["mcc"],
                 model_perf["recall"], model_perf["precision"], model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write(
            "#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])
        f.write("\n######confusion matrix######\n")
        for i in range(len(model_perf["confusion_matrix"])):
            f.write(str(model_perf["confusion_matrix"][i]) + '\n')


# Random seed
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.set_random_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(2023)

######################### Data Preprocessing #########################
print("`````````` Data Preprocessing ``````````")
# 针对sample跑循环（在每个sample上训练的组织特异性模型，在别的模型的每个sample里跑）
# set parameter
tissue_train = 'whole_blood'
tissue_test = 'kidney'
sample_num_train = 1
sample_num_test = 1
out_csv = pd.DataFrame(columns=['ACC', 'AUROC', 'AUPRC', 'F1', 'MCC', 'Recall', 'Precision', 'Specificity'])

for i in range(1, sample_num_train+1):
    # Load Model
    model = keras.models.load_model('/demo/cross_tissue/' + tissue_train + '/sample%d_DM/' % i)
    # Load sequence
    sequence_test = np.load('/demo/cross_tissue/hg19_414bpseq_test_CpG_array.npy')
    sequence_test = np.reshape(sequence_test, [-1, 414, 4, 1])
    # Load annotation(分组织)
    annotation_test = pd.read_csv('/demo/cross_tissue/' + tissue_test + '/annotation_test.csv', index_col=0)
    annotation_test = annotation_test.iloc[:, 4:]

    # preprocessing
    # Normolization(list 15 16 18 19 20)
    annotation_test.iloc[:, 15:21] = annotation_test.iloc[:, 15:21].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # delete Neighbour features
    annotation_test = pd.concat([annotation_test.iloc[:, 0:14], annotation_test.iloc[:, 20:]], axis=1)
    annotation_test = annotation_test.to_numpy()

    for j in range(1, sample_num_test+1):
        # Load label(分组织分个体)
        label_test = np.load(
            '/demo/cross_tissue/' + tissue_test + '/sample%d_label_test.npy' % j)

        print("`````````` Finish ``````````")
        ######################### Finish #########################

        ######################### DM Model Evaluating #########################
        print("`````````` Model Evaluating ``````````")
        test_scores = model.evaluate([sequence_test, annotation_test], label_test, batch_size=128)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

        y_pred = model.predict([sequence_test, annotation_test])
        y_predclass = tf.argmax(y_pred, axis=1)
        y_trueclass = tf.argmax(label_test, axis=1)
        y_prob = y_pred[:, 1]
        model_perf = evaluate_performance(y_trueclass, y_predclass, y_prob)
        '''
        np.save('./' + tissue_train + '_' + tissue_test + '/DM_train%d_test%d_y_pred.npy' % (i, j), y_predclass)
        np.save('./' + tissue_train + '_' + tissue_test + '/DM_train%d_test%d_y_test.npy' % (i, j), y_trueclass)
        np.save('./' + tissue_train + '_' + tissue_test + '/DM_train%d_test%d_y_prob.npy' % (i, j), y_prob)
        '''

        # collect each pair data in one csv
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'ACC'] = model_perf['accuracy']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'AUROC'] = model_perf['auroc']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'AUPRC'] = model_perf['auprc']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'F1'] = model_perf['f1']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'MCC'] = model_perf['mcc']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'Recall'] = model_perf['recall']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'Precision'] = model_perf['precision']
        out_csv.loc['%s%d_%s%d' % (tissue_train, i, tissue_test, j), 'Specificity'] = model_perf['specificity']

        print("`````````` Finish ``````````")
        ######################### Finish #######################

output_final = out_csv.T
output_final.to_csv('./' + tissue_train + '_' + tissue_test + '/DM_Evaluate_Result_TestSet.csv')

