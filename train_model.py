# =============================================================================
# 
# Train the model
# 
# =============================================================================


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(subX, subY, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_test, y_test)


# fixing overfitting

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 10, 20, 50, 100]}

grid = GridSearchCV(RandomForestClassifier(n_estimators=1000, 
                                           n_jobs=-1, 
                                           max_features=10),
        param_grid=param_grid, cv=5)
                                            
        
grid.fit(X_train, y_train)


grid.score(X_train, y_train)
grid.score(X_test, y_test)


#balancing, params..

print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))