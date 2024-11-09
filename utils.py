import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics
from mlxtend.plotting import plot_confusion_matrix


def add_labels(df):
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


def drop_extra_label(df_train, df_test, labels):
    for label in labels:
        df_train.drop(label, axis=1, inplace=True)
        df_test.drop(label, axis=1, inplace=True)

    return pd.concat([df_train, df_test], axis=0)


def preprocess_labels(df):
    df.drop("level", axis=1, inplace=True)

    # 多分类 40
    dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm",
                   "processtable", "mailbomb", "worm"]
    r2l_attacks = ["snmpguess", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write",
                   "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack"]
    u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps"]
    probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]
    classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
    # label2class = {0: 'Dos', 1: 'Normal', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

    def label2attack(row):
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


def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


def log_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    df[name] = np.log10(1 + x)


def data_preprocess_nsl(df):
    # 处理标签
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


def data_preprocess_unsw(df):
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

    # 将所有字符型特征进行onehot encoding
    return pd.get_dummies(df, columns=['proto', 'state', 'service'])


def data_preprocess_cic(df):
    # 将infinity替换为Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    # 'Heartbleed'类别数量过于稀少，故删除此类别
    df = df.drop(df[df.Label == 'Heartbleed'].index)

    for i in range(0, len(df.columns.values) - 1):
        if np.max(df[df.columns.values[i]]) > 10 and np.min(df[df.columns.values[i]] > -1):
            log_norm(df, df.columns.values[i])
        else:
            min_max_norm(df, df.columns.values[i])

    return df


def plot_confusing_matrix(y_true, y_pred, n_categories, outcome_labels):
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
