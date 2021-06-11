import numpy as np
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
import Model_Building as mb
import matplotlib.pyplot as plt
import tqdm

# Tuning
# choix des n_estimateurs par validation croisée
rfs = {}
for k in [10, 20, 50, 70, 100, 120, 150, 200]:
    rf=ensemble.RandomForestRegressor(n_estimators=k, oob_score=True)
    rf.fit(mb.X,mb.y)
    rfs[k] = np.mean(cross_val_score(rf, mb.X_train, mb.Y_train, cv=5))

x_plot = list(rfs.keys())
y_plot = list(rfs.values())

f, ax = plt.subplots()
ax.scatter(x_plot, y_plot)
ax.set_title("Score de validation croisée en fonction du nombre d'estimateurs")
ax.set_xlabel("Nombre d'estimateurs")
ax.set_ylabel('Score de Validation croisée')

#min leaf
rfs2 = {}
for k in tqdm(list(range(1, 11, 2))+list(range(11,25,4))):
    rf = ensemble.RandomForestRegressor(n_estimators=120, oob_score=True, min_samples_leaf=k)
    rf.fit(mb.X,mb.y)
    rfs2[k] = rf.oob_score_

x_plot = list(rfs2.keys())
y_plot = list(rfs2.values())

f, ax = plt.subplots()
ax.scatter(x_plot, y_plot)
ax.set_title()
ax.set_xlabel("Minimum d'échantillons par feuille")
ax.set_ylabel('OOB score')

# feature max
rfs2 = {}
for k in ["log2", "auto", "sqrt", 0.2, 0.1, 0.3] :
    rf = ensemble.RandomForestRegressor(n_estimators=120, oob_score=True, min_samples_leaf= 1, max_features = k)
    rf.fit(mb.X,mb.y)
    rfs2[k] = rf.oob_score_

x_plot = range(len(rfs2))  # list(rfs2.keys())
y_plot = list(rfs2.values())
print(list(rfs2.keys()))

f, ax = plt.subplots()
ax.scatter(x_plot, y_plot)
ax.set_title("Variation du svc % Minimum d'échantillons par feuille")
ax.set_xlabel("Minimum d'échantillons par feuille")
ax.set_ylabel("Cross Validation score")