import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# iris = load_iris()
# iris_data = iris.data
# iris_label = iris.target

# df_iris = pd.DataFrame(data=iris_data, columns=iris.feature_names)
# # df_iris["label"] = iris.target

# x_train, x_test, y_train, y_test = train_test_split(
#     iris_data, iris_label, test_size=0.2, random_state=11
# )

# dt_clf = DecisionTreeClassifier(random_state=11)
# dt_clf.fit(x_train, y_train)

# pred = dt_clf.predict(x_test)
# print("accuracy: {} ".format(accuracy_score(y_test, pred)))


## kfold cross validation
# iris = load_iris()
# features = iris.data
# label = iris.target

# kfold = KFold(n_splits=5)
# fold_index = 0

# dt_clf = DecisionTreeClassifier(random_state=156)

# cv_accuracy = []

# for train_index, test_index in kfold.split(features):
#     fold_index += 1

#     x_train, x_test = features[train_index], features[test_index]
#     y_train, y_test = label[train_index], label[test_index]

#     dt_clf.fit(x_train, y_train)
#     pred = dt_clf.predict(x_test)

#     accuracy = np.round(accuracy_score(y_test, pred), 4)
#     train_size = x_train.shape[0]
#     test_size = x_test.shape[0]
#     print(x_train.shape, x_test.shape, end=" ")
#     print(
#         "\n#{0} fold accuracy : {1}, train_size: {2}, val_size: {3}".format(
#             fold_index, accuracy, train_size, test_size
#         )
#     )
#     print("#{0} val index:{1}".format(fold_index, test_index))

#     cv_accuracy.append(accuracy)

# av_cvac = np.mean(cv_accuracy)
# print(av_cvac)

## Stratified K-Fold Cross Validation
## Stratified K-Fold Cross Validation
## Stratified K-Fold Cross Validation
iris = load_iris()

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris["label"] = iris.target

kfold = KFold(n_splits=3)
fold_index = 0

for train_index, test_index in kfold.split(df_iris):
    fold_index += 1
    label_train = df_iris["label"].iloc[train_index]
    label_test = df_iris["label"].iloc[test_index]

    print("## Cross validation: {0}".format(fold_index))

    print("Train label distribution:")
    print(label_train.value_counts(), end="\n\n")

    print("Val label distributioin:")
    print(label_test.value_counts(), end="\n\n")

