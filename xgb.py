import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import xgboost as xgb

train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

X = pd.concat((train[[col for col in train.columns if col.startswith("feat")]], test))
count_lk = {col: X.groupby(col).aggregate({"id": "count"}).to_dict()["id"] for col in X.columns}

y_lk = {t: i for i, t in enumerate(train.target.unique())}
y = np.array([y_lk[t] for t in train.pop("target")])
ids = train.pop("id")
test_ids = test.pop("id")
for col in train.columns:
    train[col + "_log"] = [np.log(x) if x else 0 for x in train[col]]
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    test[col + "_log"] = [np.log(x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

print(train.head())


param = {'max_depth': 3, 'eta': 0.05, 'silent':1, 'objective':'multi:softprob',
         'nthread': 4, 'eval_metric': 'mlogloss', 'seed': 1979, 'num_class': 9 }

num_round = 150

dtrain = xgb.DMatrix(np.array(train), label=y)
dtest = xgb.DMatrix(np.array(test))
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

with open("submit_xgb_one.csv", "w") as _o:
    print("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9", file=_o)
    for i in range(preds.shape[0]):
        id_ = i + 1
        row = ['{}'.format(id_)] + ["{}".format(x) for x in preds[i, :]]
        print(",".join(row), file=_o)
        

