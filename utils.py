import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics
from mlxtend.plotting import plot_confusion_matrix

# 为数据框的列添加标签
def add_labels(df):
    """
    此函数用于给输入的数据框df的列添加特定的标签。

    :param df: 输入的数据框，其列原本可能没有明确的标签或者需要重新定义标签。
    :return: 添加好标签后的数据框df。
    """
    df.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                  "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                  "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                  "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                  "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                  "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                  "attack_cat", "level"]

    return df


# 删除数据框中指定的额外标签列
def drop_extra_label(df_train, df_test, labels):
    """
    此函数用于从训练集df_train和测试集df_test中删除指定的标签列。

    :param df_train: 训练数据集的数据框。
    :param df_test: 测试数据集的数据框。
    :param labels: 要删除的标签列的列表。
    :return: 将处理后的训练集和测试集按行方向拼接后的数据框。
    """
    for label in labels:
        df_train.drop(label, axis=1, inplace=True)
        df_test.drop(label, axis=1, inplace=True)

    return pd.concat([df_train, df_test], axis=0)


# 对标签进行预处理，将攻击类别进行分类整合
def preprocess_labels(df):
    """
    此函数用于对输入数据框df中的标签进行预处理，主要是将具体的攻击类别归为几个大类。

    :param df: 包含原始标签的数据框。
    :return: 处理好标签后的数据框。
    """
    df.drop("level", axis=1, inplace=True)

    # 定义各类攻击的列表
    dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm",
                   "processtable", "mailbomb", "worm"]
    r2l_attacks = ["snmpguess", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write",
                   "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack"]
    u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps"]
    probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]
    classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
    # label2class = {0: 'Dos', 1: 'Normal', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

    def label2attack(row):
        """
        内部函数，用于根据每行数据的攻击类别将其映射到对应的大类。

        :param row: 数据框中的一行数据。
        :return: 对应的大类标签。
        """
        if row["attack_cat"] in dos_attacks:
            return classes[1]
        if row["attack_cat"] in r2l_attacks:
            return classes[2]
        if row["attack_cat"] in u2r_attacks:
            return classes[3]
        if row["attack_cat"] in probe_attacks:
            return classes[4]

        return classes[0]

    df['attack_cat'] = df.apply(label2attack, axis=1)

    return df


# 对指定列进行最小最大归一化
def min_max_norm(df, name):
    """
    此函数用于对数据框df中指定名称name的列进行最小最大归一化处理。

    :param df: 输入的数据框。
    :param name: 要进行归一化处理的列名。
    :return: 处理后的数据框（原数据框中指定列已被更新为归一化后的值）。
    """
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


# 对指定列进行对数归一化
def log_norm(df, name):
    """
    此函数用于对数据框df中指定名称name的列进行对数归一化处理。

    :param df: 输入的数据框。
    :param name: 要进行归一化处理的列名。
    :return: 处理后的数据框（原数据框中指定列已被更新为对数归一化后的值）。
    """
    x = df[name].values.reshape(-1, 1)
    df[name] = np.log10(1 + x)


# 对NSL数据集进行数据预处理
def data_preprocess_nsl(df):
    """
    此函数专门用于对NSL数据集的数据框df进行一系列的数据预处理操作。

    :param df: 输入的NSL数据集的数据框。
    :return: 处理好的数据框。
    """
    # 先处理标签
    df = preprocess_labels(df)

    traincols = list(df.columns.values)
    traincols.pop(traincols.index('protocol_type'))
    traincols.pop(traincols.index('flag'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['protocol_type', 'flag', 'service']]

    for i in range(0, len(df.columns.values) - 4):
        if np.max(df[df.columns.values[i]]) < 10:
            min_max_norm(df, df.columns.values[i])
        else:
            log_norm(df, df.columns.values[i])

    for col in ['protocol_type', 'flag', 'service', 'attack_cat']:
        df[col] = preprocessing.LabelEncoder().fit_transform(df[col])

    return df


# 对UNSW数据集进行数据预处理
def data_preprocess_unsw(df):
    """
    此函数专门用于对UNSW数据集的数据框df进行一系列的数据预处理操作。

    :param df: 输入的UNSW数据集的数据框。
    :return: 处理好的数据框。
    """
    # 将proto、state、service、label移到最后几列
    traincols = list(df.columns.values)
    traincols.pop(traincols.index('proto'))
    traincols.pop(traincols.index('state'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['proto', 'state', 'service']]

    for i in range(0, len(df.columns.values) - 3):
        if np.max(df[df.columns.values[i]]) < 10:
            min_max_norm(df, df.columns.values[i])
        else:
            log_norm(df, df.columns.values[i])

    # 将所有字符型特征进行onehot编码
    return pd.get_dummies(df, columns=['proto', 'state', 'service'])


# 对CIC数据集进行数据预处理
def data_preprocess_cic(df):
    """
    此函数专门用于对CIC数据集的数据框df进行一系列的数据预处理操作。

    :param df: 输入的CIC数据集的数据框。
    :return: 处理好的数据框。
    """
    # 将infinity替换为Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    # 'Heartbleed'类别数量过于稀少，故删除此类别
    df = df.drop(df[df.Label == 'Heartbleed'].index)

    for i in range(0, 2024-11-11):
        if np.max(df[df.columns.values[i]]) > 10 and np.min(df[df.columns.values[i]] > -1):
            log_norm(df, df.columns.values[i])
        else:
            min_max_norm(df, df.columns.values[i])

    return df


# 绘制混淆矩阵
def plot_confusing_matrix(y_true, y_pred, n_categories, outcome_labels):
    """
    此函数用于绘制混淆矩阵。

    :param y_true: 真实的标签值。
    :param y_pred: 预测的标签值。
    :param n_categories: 类别数量。
    :param outcome_labels: 各类别的标签名称。
    :return: 无（直接在图形界面展示绘制好的混淆矩阵）。
    """
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_categories)))
    plot_confusion_matrix(conf_mat=cm, class_names=outcome_labels, figsize=(10, 10), show_normed=True)
    plt.title('Confusing Matrix')
    plt.ylabel('Target')
    plt.xlabel('Predicted')
    plt.show()


if __name__ == "__main__":
    df_train = pd.read_csv(r'dataset\NSL-KDD\KDDTrain+.txt', header=None)
    df_test = pd.read_csv(r'dataset\NSL-KDD\KDDTest+.txt', header=None)
    df_train = add_labels(df_train)
    df_test = add_labels(df_test)
    # df = pd.concat([df_train, df_test])
    df_train = data_preprocess_nsl(df_train)
    df_test = data_preprocess_nsl(df_test)
    from collections import Counter
    # 0: Dos  1: Normal  2: Probe  3: R2L  4: U2R
    print(Counter(df_train['attack_cat'].values))
    print(Counter(df_test['attack_cat'].values))
