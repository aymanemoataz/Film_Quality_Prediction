import Model_Building as mb
from sklearn import ensemble
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score

rf=ensemble.RandomForestRegressor(n_estimators=500,oob_score=True, )
rf.fit(mb.X,mb.y)
print ("Training Score RandomForest: ", str(rf.score(mb.X,mb.y)))
print ("Cross Validation (10 fold) Score: " , np.mean(cross_val_score(rf, mb.X_train, mb.Y_train, cv=10)))

# Most 
top_k = 15
plt.figure(figsize=(20,8))
names = mb.X_train.columns[np.argsort(rf.feature_importances_)[::-1][:top_k]]
sns.set_theme(style="whitegrid")
values = np.sort(rf.feature_importances_)[::-1][:top_k]
plot = sns.barplot(x = names, y = values, order=names, linewidth=2.5, facecolor=(1, 1, 1, 0),
                 errcolor="1", edgecolor=".2")
_ = plot.set_xticklabels(names, rotation=15)
_ = plot.set_title('Statistiques Descriptives')
plt.savefig("StatistiquesDescriptives2.png")
plt.show()
