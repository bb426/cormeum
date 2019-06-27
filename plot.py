# -*- coding: utf-8 -*-
"""
academic paper 용 random forest model 비교
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%% Load data

X_train, X_test, y_train, y_test = train_test_split(subX, subY, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
print('train accuracy without configuring hyperparameters: {:.3f}'.format(model.score(X_train, y_train)))
print('test accuracy without configuring hyperparameters: {:.3f}'.format(model.score(X_test, y_test)))


#%% Accuracy score without configuring hyperparameters
result = []
for i in range(1, 101):
    model = RandomForestClassifier(n_estimators=i, n_jobs=-1)
    model.fit(X_train, y_train)
    result.append([model.score(X_train, y_train), model.score(X_test, y_test)])
    
result_array = np.array(result)
result_array[:,0] #train_accu
result_array[:,1] #test_accu

n_estimator = range(1, 101)
plt.plot(n_estimator, result_array[:, 0], 'bo', label='Training accuracy')
plt.plot(n_estimator, result_array[:, 1], 'b', label='Test accuracy')
plt.xlabel('n_estimator')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#%% Accuracy score with configuring hyperparameters

result_grid = []
for max_depth in [1, 2, 5, 10, 20, 50, 100]:
    for max_features in [1, 2, 5, 10, 30, 50]:
        model = RandomForestClassifier(n_estimators=i, n_jobs=-1, max_depth=max_depth, max_features=max_features)
        model.fit(X_train, y_train)
        result_grid.append(model.score(X_test, y_test))
        
result_grid_array = np.array(result_grid)
result_grid_array = result_grid_array.reshape(7, 6)
result_grid_df = pd.DataFrame(result_grid_array, columns=[1, 2, 5, 10, 30, 50], index=[1, 2, 5, 10, 20, 50, 100])

sns.heatmap(result_grid_df[::-1], annot=True)
plt.xlabel('max features')
plt.ylabel('max depth')
plt.show()

