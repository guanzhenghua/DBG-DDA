import rvfl
import numpy as np
from rvfl import rvfl_train
import os
import time
from scipy.io import savemat

def roc1(predict_label, test_data_label):
    """
    计算分类器评估指标
    predict_label: 分类器对于每个样本的预测标签，行或列向量
    test_data_label: 样本的真实标签，行或列向量
    """
    predict_label = predict_label.flatten()
    test_data_label = test_data_label.flatten()
    l = len(predict_label)
    TruePositive = 0
    TrueNegative = 0
    FalsePositive = 0
    FalseNegative = 0

    if np.min(predict_label) == 0:
        predict_label[predict_label == 0] = -1
        test_data_label[test_data_label == 0] = -1

    for k in range(l):
        if test_data_label[k] == 1 and predict_label[k] == 1:  # 真阳性
            TruePositive += 1
        if test_data_label[k] == -1 and predict_label[k] == -1:  # 真阴性
            TrueNegative += 1
        if test_data_label[k] == -1 and predict_label[k] == 1:  # 假阳性
            FalsePositive += 1

        if test_data_label[k] == 1 and predict_label[k] == -1:  # 假阴性
            FalseNegative += 1
    if TruePositive + TrueNegative + FalsePositive + FalseNegative == l:
        ACC = (TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative)
        SN = TruePositive / (TruePositive + FalseNegative)
        SP = TrueNegative / (TrueNegative + FalsePositive)
        PPV = TruePositive / (TruePositive + FalsePositive)
        NPV = TrueNegative / (TrueNegative + FalseNegative)
        F1 = 2 * ((SN * PPV) / (SN + PPV))
        MCC = (TruePositive * TrueNegative - FalsePositive * FalseNegative) / np.sqrt(
            (TruePositive + FalseNegative) * (TrueNegative + FalsePositive) * (TruePositive + FalsePositive) * (
                        TrueNegative + FalseNegative))

    return ACC, SN, SP, PPV, NPV, F1, MCC

def auc(test_targets, output):
    """
    计算AUC值
    test_targets: 原始样本标签，行或列向量
    output: 分类器得到的判为正类的概率，行或列向量
    """
    I = np.argsort(output)
    M = np.sum(test_targets == 1)  # 正类样本数
    N = np.sum(test_targets == 0)  # 负类样本数
    sigma = 0
    for i in range(M + N - 1, -1, -1):
        if test_targets[I[i]] == 1:
            sigma += i + 1  # 正类样本rank相加
    result = (sigma - (M + 1) * M / 2) / (M * N)
    return result


option = {'Scale': 1, 'Scalemode': 1, 'bias': 1, 'link': 1}

i = 0
dataset_path = 'D:/myPorject/FuHLDR-main/data/B-Dataset/'
Yeast_f1_train_feature = np.loadtxt(dataset_path + 'train_data_' + str(i) + '.csv', delimiter=',')
Yeast_f1_train_label = np.loadtxt(dataset_path + 'train_labels_' + str(i) + '.csv', delimiter=',')
Yeast_f1_test_feature = np.loadtxt(dataset_path + 'test_data_' + str(i) + '.csv', delimiter=',')
Yeast_f1_test_label = np.loadtxt(dataset_path + 'test_labels_' + str(i) + '.csv', delimiter=',')


N_list = [10, 11, 12, 13, 14, 15, 16]
C_list = [3, 6, 12]
best_acc = 0.0
option_best = {}

for i in range(len(N_list)):
    for j in range(len(C_list)):
        option = {}
        option['N'] = 2 ** N_list[i]
        option['C'] = 2 ** C_list[j]
        # time
        tic = time.time()
        predictions_f1, TrainingAccuracy_f1, TestingAccuracy_f1 = rvfl_train(Yeast_f1_train_feature, Yeast_f1_train_label, Yeast_f1_test_feature, Yeast_f1_test_label, option)
        ACC, SN, SP, PPV, NPV, F1, MCC = roc1(predictions_f1, Yeast_f1_test_label)
        AUC = auc(predictions_f1, Yeast_f1_test_label)
        toc = time.time()
        during = toc - tic
        aa = [ACC, SN, SP, PPV, NPV, F1, MCC, AUC]
        print("ACC:", ACC)
        final = {'TrainingAccuracy_f1': TrainingAccuracy_f1,
                 'TestingAccuracy_f1': TestingAccuracy_f1, 'ACC': ACC, 'SN': SN, 'SP': SP, 'PPV': PPV, 'NPV': NPV,
                 'F1': F1,
                 'MCC': MCC, 'AUC': AUC}
        print(final)
        print("Execution time: {:.2f} seconds".format(during))
        if ACC > best_acc:
            best_acc = ACC
            option_best = option
        # save parameters and results
        mdx = 'D:/myPorject/FuHLDR-main/data/B-Dataset/pyresults'
        if not os.path.exists(mdx):
            os.makedirs(mdx)
        SavePathName = os.path.join(mdx, 'Results_N_' + str(N_list[i]) + '_C_' + str(option['C']) + '.mat')
        savemat(SavePathName, {'predictions_f1': predictions_f1, 'TrainingAccuracy_f1': TrainingAccuracy_f1, 'TestingAccuracy_f1': TestingAccuracy_f1, 'ACC': ACC, 'SN': SN, 'SP': SP, 'PPV': PPV, 'NPV': NPV, 'F1': F1, 'MCC': MCC, 'AUC': AUC, 'time': during})
print(f"best_acc {best_acc}\n")

n_fold = 9
time = []
for i in range(n_fold + 1):
    print(f"fold {i}")
    # load data
    dataset_path = 'D:/myPorject/FuHLDR-main/data/B-Dataset/'
    Yeast_f1_train_feature = np.loadtxt(f"{dataset_path}train_data_{i}.csv", delimiter=",")
    Yeast_f1_train_label = np.loadtxt(f"{dataset_path}train_labels_{i}.csv", delimiter=",")
    Yeast_f1_test_feature = np.loadtxt(f"{dataset_path}test_data_{i}.csv", delimiter=",")
    Yeast_f1_test_label = np.loadtxt(f"{dataset_path}test_labels_{i}.csv", delimiter=",")
    # time
    # tic = time.monotonic()
    # training
    predictions_f1, TrainingAccuracy_f1, TestingAccuracy_f1 = rvfl_train(
        Yeast_f1_train_feature, Yeast_f1_train_label, Yeast_f1_test_feature, Yeast_f1_test_label, option_best)
    ACC, SN, SP, PPV, NPV, F1, MCC = roc1(predictions_f1, Yeast_f1_test_label)
    AUC = auc(predictions_f1, Yeast_f1_test_label)
    final = {'TrainingAccuracy_f1': TrainingAccuracy_f1,
     'TestingAccuracy_f1': TestingAccuracy_f1, 'ACC': ACC, 'SN': SN, 'SP': SP, 'PPV': PPV, 'NPV': NPV, 'F1': F1,
     'MCC': MCC, 'AUC': AUC}
    print(final)
    # time.append(time.monotonic() - tic)
    # save results of the model 10-CV
    outputID = os.path.join(mdx, f"{i}.txt")
    np.savetxt(outputID, predictions_f1, delimiter="\t")


