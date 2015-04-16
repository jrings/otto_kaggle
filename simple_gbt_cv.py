import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

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
    train[col + "_count"] = ([np.log(x) if x else 0 for x in train[col + "_count"]])
    test[col + "_log"] = [np.log(x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])    
    test[col + "_count"] = ([np.log(x) if x else 0 for x in test[col + "_count"]])

print(train.head())

model = GradientBoostingClassifier(random_state=1979, min_samples_split=7,
                             max_depth=4, max_features="sqrt", n_estimators=800)
print(np.mean(cross_val_score(model, np.array(train), y, scoring="log_loss", n_jobs=4, cv=4)))

model.fit(np.array(train), y)
preds = model.predict_proba(np.array(test))

df = pd.DataFrame({"Class_{}".format(i+1): preds[:, i] for i in range(preds.shape[1])})
df["id"] = test_ids

df = df[["id"] + ["Class_{}".format(i+1) for i in range(preds.shape[1])]]
df.to_csv("submit_simple.csv", index=False, float_format="%.5f")



