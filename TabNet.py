import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

from utils import add_labels, preprocess_labels, drop_extra_label

# 设置训练集和测试集的CSV文件路径
# 这里最初是NSL-KDD数据集的路径，但后来被注释掉并替换为UNSW_NB15数据集的路径
# train_csv_path = r'dataset\NSL-KDD\KDDTrain+.txt'
# test_csv_path = r'dataset\NSL-KDD\KDDTest+.txt'

train_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_training-set.csv'
test_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_testing-set.csv'

# 读取训练集和测试集数据，这里最初读取时设置header=None，但后来被注释掉，说明数据本身有表头
# df_train = pd.read_csv(train_csv_path, header=None)
# df_test = pd.read_csv(test_csv_path, header=None)
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# 调用函数drop_extra_label，从训练集和测试集数据中删除指定的列（'id'和'label'），并将处理后的结果保存在df_X中
df_X = drop_extra_label(df_train, df_test, ['id', 'label'])

# 以下两行代码被注释掉，说明原本可能有不同的数据处理方式，但现在采用了下面的处理流程
# df_X = pd.concat([add_labels(df_train), add_labels(df_test)], axis=0)
# df_X = preprocess_labels(df_X)

# 对df_X中的'proto'、'state'、'service'列进行标签编码，将其转换为数值形式，以便后续模型处理
for col in ['proto', 'state', 'service']:
    df_X[col] = LabelEncoder().fit_transform(df_X[col])

# 从df_X中弹出'attack_cat'列作为目标变量，对其进行标签编码，并将编码后的结果保存在df_Y中
df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat').values)

# 将数据集划分为训练集、测试集，其中测试集占比为0.25，设置随机种子为666以确保划分结果可复现
x_train, x_test, y_train, y_test = train_test_split(df_X.values, df_Y, test_size=0.25, random_state=666)
# 进一步将训练集划分为训练集和验证集，其中验证集占训练集的0.25，同样设置随机种子为666
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=666)

# 创建TabNetClassifier模型实例，设置优化器为Adam，学习率为2e-3，以及其他相关参数
tabnet = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                          optimizer_params=dict(lr=2e-3),
                          # scheduler_params={"step_size": 10, "gamma": 0.9},
                          # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                          mask_type='entmax'
                          )

# 使用训练集和验证集对TabNet模型进行训练，设置评估指标、训练轮数、耐心值等参数
tabnet.fit(x_train, y_train,
           eval_name=['train', 'valid'],
           eval_set=[(x_train, y_train), (x_val, y_val)],
           eval_metric=['accuracy'],
           max_epochs=50, patience=50,
           batch_size=512, virtual_batch_size=256,
           num_workers=0, weights=1,
           drop_last=False
           )

# 使用训练好的TabNet模型对测试集进行预测，得到预测结果
y_pred = tabnet.predict(x_test)

# 计算预测结果的准确率，通过比较真实标签y_test和预测标签y_pred得出
acc = accuracy_score(y_test, y_pred)
# 计算预测结果的精确率，采用加权平均方式，通过比较真实标签y_test和预测标签y_pred得出
precision = precision_score(y_test, y_pred, average='weighted')
# 计算预测结果的召回率，采用加权平均方式，通过比较真实标签y_test和预测标签y_pred得出
recall = recall_score(y_test, y_pred, average='weighted')
# 计算预测结果的F1值，采用加权平均方式，通过比较真实标签y_test和预测标签y_pred得出
f1 = f1_score(y_test, y_pred, average='weighted')

# 打印准确率、精确率、召回率和F1值的结果
# 这里注释掉了之前的两组结果示例，可能是之前不同数据集或模型设置下得到的结果
# accuracy is 0.9329235163538707, precision is 0.9361260787842635, recall is 0.9329235163538707, f1 score is 0.9334960049480963
# accuracy is 0.9749798007002424, precision is 0.9831367450374873, precision is 0.9749798007002424, f1 score is 0.9784177021252124
print("accuracy is {0}, precision is {1}, recall is {2}, f1 score is {3}".format(acc, precision, recall, f1))

# 绘制模型训练过程中的损失值变化曲线
plt.figure(1)
plt.plot(tabnet.history['loss'])
plt.show()

# 绘制模型训练过程中训练集和验证集的准确率变化曲线
plt.figure(2)
plt.plot(tabnet.history['train_accuracy'])
plt.plot(tabnet.history['valid_accuracy'])
plt.show()
