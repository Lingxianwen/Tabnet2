import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

from utils import add_labels, preprocess_labels, drop_extra_label

# train_csv_path = r'dataset\NSL-KDD\KDDTrain+.txt'
# test_csv_path = r'dataset\NSL-KDD\KDDTest+.txt'

train_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_training-set.csv'
test_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_testing-set.csv'

# df_train = pd.read_csv(train_csv_path, header=None)
# df_test = pd.read_csv(test_csv_path, header=None)
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)
df_X = drop_extra_label(df_train, df_test, ['id', 'label'])
# df_X = pd.concat([add_labels(df_train), add_labels(df_test)], axis=0)
# df_X = preprocess_labels(df_X)
for col in ['proto', 'state', 'service']:
    df_X[col] = LabelEncoder().fit_transform(df_X[col])

df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat').values)

x_train, x_test, y_train, y_test = train_test_split(df_X.values, df_Y, test_size=0.25, random_state=666)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=666)

tabnet = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                          optimizer_params=dict(lr=2e-3),
                          # scheduler_params={"step_size": 10, "gamma": 0.9},
                          # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                          mask_type='entmax'
                          )

tabnet.fit(x_train, y_train,
           eval_name=['train', 'valid'],
           eval_set=[(x_train, y_train), (x_val, y_val)],
           eval_metric=['accuracy'],
           max_epochs=50, patience=50,
           batch_size=512, virtual_batch_size=256,
           num_workers=0, weights=1,
           drop_last=False
           )


y_pred = tabnet.predict(x_test)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# accuracy is 0.9329235163538707, precision is 0.9361260787842635, recall is 0.9329235163538707, f1 score is 0.9334960049480963
# accuracy is 0.9749798007002424, precision is 0.9831367450374873, recall is 0.9749798007002424, f1 score is 0.9784177021252124
print("accuracy is {0}, precision is {1}, recall is {2}, f1 score is {3}".format(acc, precision, recall, f1))

plt.figure(1)
plt.plot(tabnet.history['loss'])
plt.show()
plt.figure(2)
plt.plot(tabnet.history['train_accuracy'])
plt.plot(tabnet.history['valid_accuracy'])
plt.show()
