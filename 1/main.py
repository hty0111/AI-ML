import joblib
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from minepy import MINE
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from IPython.display import display
from datetime import datetime as dt
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')


def data_split(features, label):
    """
    数据切分
    :param features: 特征选择后的输入特征数据
    :param label: 标签数据
    :return: X_train:数据切分后的训练数据
             X_val:数据切分后的验证数据
             X_test:数据切分后的测试数据
             y_train:数据切分后的训练数据标签
             y_val:数据切分后的验证数据标签
             y_test:数据切分后的测试数据标签
    """
    # 将 features 和 label 数据切分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0, stratify=label)

    # 将 X_train 和 y_train 进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_predict(learner, X_train, y_train, X_val, y_val):
    '''
    模型训练验证
    :param learner: 监督学习模型
    :param X_train: 训练集 特征数据
    :param y_train: 训练集 类标
    :param X_val: 验证集 特征数据
    :param y_val: 验证集 类标
    :return: results: 训练与验证结果
    '''

    results = {}

    # 使用训练集数据来拟合学习器
    start = time()  # 获得程序开始时间
    learner = learner.fit(X_train, y_train)
    end = time()  # 获得程序结束时间

    # 计算训练时间
    # results['train_time'] = end - start

    # 得到在验证集上的预测值
    start = time()  # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train)
    end = time()  # 获得程序结束时间

    # 计算预测用时
    # results['pred_time'] = end - start

    # 计算在训练数据的准确率
    results['acc_train'] = round(accuracy_score(y_train, predictions_train),4)

    # 计算在验证上的准确率
    results['acc_val'] = round(accuracy_score(y_val, predictions_val),4)

    # 计算在训练数据上的召回率
    results['recall_train'] = round(recall_score(y_train, predictions_train),4)

    # 计算验证集上的召回率
    results['recall_val'] = round(recall_score(y_val, predictions_val),4)

    # 计算在训练数据上的F-score
    results['f_train'] = round(fbeta_score(y_train, predictions_train, beta=1),4)

    # 计算验证集上的F-score
    results['f_val'] = round(fbeta_score(y_val, predictions_val, beta=1),4)

    # 成功
    print("{} trained on {} samples.".format(learner.__class__.__name__, len(X_val)))

    # 返回结果
    return results


def plot_learning_curve(estimator, X, y, cv=None, n_jobs=1):
    """
    绘制学习曲线
    :param estimator: 训练好的模型
    :param X:绘制图像的 X 轴数据
    :param y:绘制图像的 y 轴数据
    :param cv: 交叉验证
    :param n_jobs:
    :return:
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure('Learning Curve', facecolor='lightgray')
    plt.title('Learning Curve')
    plt.xlabel('train size')
    plt.ylabel('score')
    plt.grid(linestyle=":")
    plt.plot(train_sizes, train_scores_mean, label='traning score')
    plt.plot(train_sizes, test_scores_mean, label='val score')
    plt.legend()
    plt.show()


def search_model(X_train, y_train, X_val, y_val, model_save_path):
    """
    创建、训练、优化和保存深度学习模型
    :param X_train, y_train: 训练集数据
    :param X_val, y_val: 验证集数据
    :param model_save_path: 保存模型的路径和名称
    """

    # 创建监督学习模型 以决策树为例
    clf = tree.DecisionTreeClassifier(random_state=42)

    # 创建调节的参数列表
    parameters = {'max_depth': range(5,10),
                  'min_samples_split': range(2,10)}

    # 创建一个fbeta_score打分对象 以F-score为例
    scorer = make_scorer(fbeta_score, beta=1)

    # 在分类器上使用网格搜索，使用'scorer'作为评价函数
    kfold = KFold(n_splits=10) #切割成十份

    # 同时传入交叉验证函数
    grid_obj = GridSearchCV(clf, parameters, cv=kfold)

    #绘制学习曲线
    plot_learning_curve(clf, X_train, y_train, cv=kfold, n_jobs=4)

    # 用训练数据拟合网格搜索对象并找到最佳参数
    grid_obj.fit(X_train, y_train)

    # 得到estimator并保存
    best_clf = grid_obj.best_estimator_
    joblib.dump(best_clf, model_save_path)

    # 使用没有调优的模型做预测
    predictions = (clf.fit(X_train, y_train)).predict(X_val)
    best_predictions = best_clf.predict(X_val)


def load_and_model_prediction(X_test,y_test,model_path):
    """
    加载模型和评估模型
    :param X_test,y_test: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    #加载模型
    my_model = joblib.load(model_path)

    #对测试数据进行预测
    copy_test = [value for value in X_test]
    copy_predicts = my_model.predict(X_test)

    print ("Accuracy on test data: {:.4f}".format(accuracy_score(y_test, copy_predicts)))
    print ("Recall on test data: {:.4f}".format(recall_score(y_test, copy_predicts)))
    print ("F-score on test data: {:.4f}".format(fbeta_score(y_test, copy_predicts, beta = 1)))


def data_processing_and_feature_selecting(data_path):
    """
    特征选择
    :param  data_path: 数据集路径
    :return: new_features,label: 经过预处理和特征选择后的特征数据、类标数据
    """
    data_xls = pd.ExcelFile('DataSet.xlsx')
    data = {}
    for name in data_xls.sheet_names:
        df = data_xls.parse(sheet_name=name, header=None)
        # print("%-8s 表的 shape:" % name, df.shape)
        data[name] = df
    # print(data)
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']
    scalar = MinMaxScaler()
    feature1 = pd.DataFrame(scalar.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scalar.fit_transform(feature2_raw))

    select_feature_number = 5
    select_feature1 = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number
                                  ).fit(feature1, np.array(label).flatten()).get_support(indices=True)

    select_feature2 = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number
                                  ).fit(feature2, np.array(label).flatten()).get_support(indices=True)
    new_features = pd.concat([feature1[feature1.columns.values[select_feature1]],
                              feature2[feature2.columns.values[select_feature2]]], axis=1)
    # 返回筛选后的数据
    return new_features, label


if __name__ == "__main__":
    # 特征提取
    new_features, label = data_processing_and_feature_selecting('DataSet.xlsx')
    # 进行数据切分
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(new_features, label)

    # 显示切分的结果
    # print("Training set has {} samples.".format(X_train.shape[0]))
    # print("Validation set has {} samples.".format(X_val.shape[0]))
    # print("Testing set has {} samples.".format(X_test.shape[0]))

    # 初始化三个模型
    clf_A = tree.DecisionTreeClassifier(random_state=42)
    clf_B = naive_bayes.GaussianNB()
    clf_C = svm.SVC()

    # 收集学习器的结果
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        results[clf_name] = train_predict(clf, X_train, y_train, X_val, y_val)

    # 打印三个模型得到的训练验证结果
    # print("高斯朴素贝叶斯模型结果:", results['GaussianNB'])
    # print("支持向量机模型结果:", results['SVC'])
    # print("决策树模型结果:", results['DecisionTreeClassifier'])

    search_model(X_train, y_train, X_val, y_val, model_save_path='./results/my_model.m')

    # 加载模型并对测试样本进行测试
    model_path = "./results/my_model.m"
    load_and_model_prediction(X_test, y_test, model_path)

