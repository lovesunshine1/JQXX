
import pandas as pd       #pandas库，用于数据读取、清洗、处理、统计和输出，数据结构有 DataFrame 和 Series
import numpy as np        #numpy库，用来进行数组矩阵操作
from sklearn.model_selection import StratifiedKFold                                 #使用sklearn的交叉验证--K层折叠
from sklearn.metrics import roc_auc_score                                           #评价指标AUC-roc曲线下面积
import warnings           #忽略警报操作，防止输出中有警报的干预，不便于查看
warnings.filterwarnings('ignore')

# 3种决策树模型
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier        #通过StackingClassifier()将上述模型进行Stacking融合
from sklearn.linear_model import LogisticRegression    #使用Stack的LogisticRegression模型
from sklearn.preprocessing import LabelEncoder         #LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码

# 1.数据读取-- #read_csv()读取csv文件数据；
train_data = pd.read_csv('dataTrain.csv')
test_data = pd.read_csv('dataA.csv')
submission = pd.read_csv('Lwt_submit.csv')

# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]] + 1)

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]] + 1)
# 离散化
all_f = [f'f{idx}' for idx in range(1, 47) if idx != 3]
for df in [train_data, test_data]:
    for col in all_f:
        df[f'{col}_log'] = df[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)
# 特征交叉
log_f = [f'f{idx}_log' for idx in range(1, 47) if idx != 3]
for df in [train_data, test_data]:
    for i in range(len(log_f)):
        for j in range(i + 1, len(log_f)):
            df[f'{log_f[i]}_{log_f[j]}'] = df[log_f[i]] * 10000 + df[log_f[j]]


#对列f3进行数值化替换操作
cat_columns = ['f3']
#concat()数据合并
data = pd.concat([train_data, test_data])
for col in cat_columns:

    #LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码
    lb = LabelEncoder()
    lb.fit(data[col])

    #transform(y) ：将y转变成索引值
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])
feature_columns = [col for col in train_data.columns if col not in ['id', 'label']]
target = 'label'

#训练集、测试集
train = train_data[feature_columns]
test = test_data[feature_columns]
label = train_data[target]


#训练和验证分离，交叉验证法-StratifiedKFold分层K折
def model_train(model, model_name, kfold=10):

    #定义训练和测试矩阵
    train_compl = np.zeros((train.shape[0]))
    test_compl = np.zeros(test.shape[0])

    #这里是采用StratifiedKFold分层10折
    skf = StratifiedKFold(n_splits=kfold)
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):

        # loc()通过行索引 Index中的具体值来取行数据
        x_train, x_test = train.loc[train_index, :], train.loc[test_index, :]
        y_train, y_test = label.loc[train_index], label.loc[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_test)[:, 1]
        train_compl[test_index] = y_pred.ravel()

        #计算每次的ACU评价指标
        auc = roc_auc_score(y_test, y_pred)
        print("KFold = %d, AUC = %.4f" % (k, auc))
        test_fold_compl = model.predict_proba(test)[:, 1]
        test_compl += test_fold_compl.ravel()

    # 输出总体的ACU评价指标
    print("General AUC = %.4f" % (roc_auc_score(label, train_compl)))
    return test_compl / kfold

gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6,
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05,
    n_estimators=200,
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

#选取3个树模型进行Stacking模型融合
estimators = [
    ('xgbc', xgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
#使用Stack的LogisticRegression模型
#通过StackingClassifier()将上述模型进行Stacking融合
SC = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 调用分层10折交叉验证 对基于决策树的融合模型 进行模型训练
compl = model_train(SC, "StackingClassifier", 10)

#提交，生成csv文件
submission['label'] = compl
submission.to_csv('Lwt_submit.csv', index=False)
