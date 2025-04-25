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
    model_perf = {"auroc": auroc, "auroc_curve": auroc_curve,
                  "auprc": auprc, "auprc_curve": auprc_curve,
                  "accuracy": accuracy, "mcc": mcc,
                  "recall": recall, "precision": precision, "f1": f1,
                  "class_report": class_report,
                  "confusion_matrix": confusion_matrix}

    return model_perf

# Output result of evaluation
def eval_output(model_perf,output_file):
    with open(output_file,'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tMCC=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
               (model_perf["auroc"],model_perf["auprc"],model_perf["accuracy"],model_perf["mcc"],model_perf["recall"],model_perf["precision"],model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write("#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])
        f.write("\n######confusion matrix######\n")
        for i in range (len(model_perf["confusion_matrix"])):
            f.write(str(model_perf["confusion_matrix"][i])+'\n')

# model architecture
class DeepMethylation(keras.models.Model):
    def __init__(self):
        # 调用父类的构造方法
        super(DeepMethylation, self).__init__()
        # 添加新属性
        '''
        define model's construction
        filters:卷积核个数(输出的通道数） kernel_size:卷积核的大小 strides:步长
        padding:是否填充原图像valid否 same activation:激活函数 input_shape:输入的图像的大小，为1通道
        '''
        # CNN部分的模型层次
        # first layer:Conv+Reshape
        self.conv1_layer = keras.layers.Conv2D(filters=16, kernel_size=[15, 4], strides=(1, 1),
                                               padding='valid', activation=None, input_shape=(414, 4, 1))
        # seconda layer:Conv+Pool
        self.conv2_layer = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=(1, 1),
                                               padding='valid', activation='relu', input_shape=(20, 20, 16))
        self.batchnormalization1_layer = keras.layers.BatchNormalization()
        self.pool2_layer = keras.layers.MaxPool2D(pool_size=(3, 3), strides=3)
        # third layer:Conv only
        self.conv3_layer = keras.layers.Conv2D(filters=48, kernel_size=[3, 3], strides=(1, 1),
                                               padding='valid', activation=None, input_shape=(6, 6, 32))
        # forth layer:Conv only
        self.conv4_layer = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                                               padding='valid', activation=None, input_shape=(4, 4, 48))
        # full connection
        self.full_connection1 = keras.layers.Dense(80, activation='relu')

        # MLP
        self.mlp_layer1 = keras.layers.Dense(16, activation='relu')
        self.batchnormalization2_layer = keras.layers.BatchNormalization()

        # integration output
        self.dropout_layer = keras.layers.Dropout(rate=0.5)
        self.output_layer = keras.layers.Dense(2, activation='softmax')

    def call(self, input):
        input1, input2 = input
        # CNN
        # first layer:Conv+Reshape
        conv1 = self.conv1_layer(input1)
        conv1_reshape = tf.reshape(conv1, [-1, 20, 20, 16])
        # seconda layer:Conv+Pool
        conv2 = self.conv2_layer(conv1_reshape)
        conv2 = self.batchnormalization1_layer(conv2)
        pool2 = self.pool2_layer(conv2)
        # third layer:Conv only
        conv3 = self.conv3_layer(pool2)
        # forth layer:Conv only
        conv4 = self.conv4_layer(conv3)
        # Flatten
        conv4_reshape = tf.keras.layers.Flatten()(conv4)
        # full connection
        fc = self.full_connection1(conv4_reshape)

        # MLP
        mlp1 = self.mlp_layer1(input2)
        mlp1 = self.batchnormalization2_layer(mlp1)

        # integration output
        concat = keras.layers.concatenate([mlp1, fc])
        dropout = self.dropout_layer(concat)
        output = self.output_layer(dropout)

        return output

#Random seed
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.set_random_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(2023)

#external validation: train on 450k data test on 850k data
######################### Data Preprocessing #########################
print("`````````` Data Preprocessing ``````````")
# Load sequence
sequence = np.load(
    '/demo/external_validation/450k_hg19_414bpseq_train_CpG_array.npy')
sequence = np.reshape(sequence, [-1, 414, 4, 1])
# Load annotation(分组织)
annotation = pd.read_csv(
    '/demo/external_validation/450k_annotation_train.csv',
    index_col=0)
annotation = annotation.iloc[:, 5:]
# Load label（分组织分个体）
label = np.load(
    '/demo/external_validation/450k_label_train.npy')

# preprocessing
# Normolization(list 15 16 18 19 20)
annotation.iloc[:, 15:21] = annotation.iloc[:, 15:21].apply(
    lambda x: (x - x.mean()) / (x.std()))

# delete Neighbour features
annotation = pd.concat([annotation.iloc[:, 0:14], annotation.iloc[:, 20:]], axis=1)
annotation = annotation.to_numpy()

# TrainingSet : ValidatingSet = 3 : 1
sequence_train, sequence_val, annotation_train, annotation_val, label_train, label_val = train_test_split(sequence,annotation,label,test_size=0.25)
print("`````````` Finish ``````````")
######################### Finish #########################

######################### DM Model Construction #########################
print("`````````` Model Construction ``````````")
label_train = tf.cast(label_train, tf.float32)
label_val = tf.cast(label_val, tf.float32)
sequence_train = tf.cast(sequence_train, tf.float32)
sequence_val = tf.cast(sequence_val, tf.float32)

model = DeepMethylation()

tf.config.run_functions_eagerly(True)
print("`````````` Finish ``````````")
######################### Finish #######################

######################### DM Model Training #########################
print("`````````` Model Training ``````````")
opt = keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 200)  # lr=lr*(1/1+decay*iter))
# loss_fn = keras.losses.binary_crossentropy(from_logits=True)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['binary_accuracy'])
# monitor：监控指标，默认val_loss min_delta:监控指标的最小提升值 patience：在监控指标没有提升时，epoch等待轮数
callbacks = [keras.callbacks.EarlyStopping(patience=3, min_delta=2e-4)]
history = model.fit([sequence_train, annotation_train], label_train,
                        batch_size=128, epochs=50,
                        validation_data=([sequence_val, annotation_val], label_val),
                        callbacks=callbacks)

# save model
#model.save('./Blood_450k_DM.h5')
model.save('/./Blood_450k_DM/', save_format='tf')
print("`````````` Finish ``````````")
######################### Finish #######################




#Evaluation
#Load 450k model to predict 850k array data
######################### Data Preprocessing #########################
print("`````````` Data Preprocessing ``````````")
# Load Model
model = keras.models.load_model('/demo/external_validation/Blood_450k_DM_model/')
# Load sequence
sequence_test = np.load('/demo/external_validation/850k_hg19_414bpseq_test_CpG_array.npy')
sequence_test = np.reshape(sequence_test, [-1, 414, 4, 1])
# Load annotation(分组织)
annotation_test = pd.read_csv('/demo/external_validation/850k_annotation_test.csv',index_col=0)
annotation_test = annotation_test.iloc[:, 4:]
# Load label（分组织分个体）
label_test = np.load('/demo/external_validation/850k_label_test.npy')

# preprocessing
# Normolization(list 15 16 18 19 20)
annotation_test.iloc[:, 15:21] = annotation_test.iloc[:, 15:21].apply(
    lambda x: (x - x.mean()) / (x.std()))
# delete Neighbour features
annotation_test = pd.concat([annotation_test.iloc[:, 0:14], annotation_test.iloc[:, 20:]], axis=1)

annotation_test = annotation_test.to_numpy()

# 筛选位点
f450k_850k_sites = pd.read_csv('/demo/external_validation/f450k_850k_sites.csv', header=0)
f850k_non_450k_sites = pd.read_csv('/demo/external_validation/f850k_non_450k_sites.csv', header=0)
'''
#450k与850k交集
sequence_test = sequence_test[f450k_850k_sites['Raw_num_x']]
annotation_test = annotation_test.iloc[f450k_850k_sites['Raw_num_x'],:]
label_test = label_test[f450k_850k_sites['Raw_num_x']]

#850k与450k补集
sequence_test = sequence_test[f850k_non_450k_sites['Raw_num']]
annotation_test = annotation_test.iloc[f850k_non_450k_sites['Raw_num'],:]
label_test = label_test[f850k_non_450k_sites['Raw_num']]
'''

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
eval_output(model_perf, './DM_Evaluate_Result_TestSet.txt')

print("`````````` Finish ``````````")
######################### Finish #########################