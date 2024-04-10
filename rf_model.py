#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
import numpy as np
import string as str
#import nltk
import networkx as nx # you might need to install this
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import pickle
import warnings
import re
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.base import TransformerMixin

# Display all columns
pd.set_option('display.max_columns', None)


#%%
# Import data
# feature engineering: added new features of averages of SA for previous five years (by month)
snow_main_fteng = pd.read_csv('snow_main_new.csv')


#%%
### Drop NAs from original dataset:
# Check for features containing null obs
# features_na = snow_main_fteng.columns[snow_main_fteng.isna().any()].tolist()
# features_na_drop = features_na[11:17]

# # snow_main_clean drops NA's in specified features
# snow_main_clean = snow_main_fteng.dropna(subset=features_na_drop) if features_na_drop else snow_main_fteng


# %%

# Remaining Feature Engineering
# Convert object features to factors

# snow_main_clean['Site_Name'] = snow_main_clean['Site_Name'].astype('category')
# snow_main_clean['County'] = snow_main_clean['County'].astype('category')

# Encode categorical features
snow_main_clean = snow_main_fteng
snow_main_clean['Apr'] = snow_main_clean['Apr'].fillna(0.001)
snow_main_clean = pd.get_dummies(snow_main_clean, columns=['Site_Name', 'County'])

# #%%
# snow_main_clean = snow_main_clean.iloc[:, 0:19]

#%%
# Matt's feature eng
months = ["Jan","Feb","Mar"]
snow_main_clean["Snow to Elevation"] = snow_main_clean[months].sum(axis=1).div(snow_main_clean["Elev"])

# y = snow_main_clean["Apr"] + .001
# y = y.fillna(0.001)
# X = snow_main_clean.drop(["Apr","Apr (WE)","May", "May (WE)", "Jun", "Jun (WE)"], axis=1)

# %%
# Parse sets
X = snow_main_clean.drop(['Apr', 'Apr (WE)', 'May', 'May (WE)', 'Jun', 'Jun (WE)', 'Jan_SA_Avg', 'Feb_SA_Avg', 'Mar_SA_Avg', 'May_SA_Avg', 'Jun_SA_Avg', 'Jan_WE_Avg', 'Feb_WE_Avg', 'Mar_WE_Avg', 'May_WE_Avg', 'Jun_WE_Avg'], axis=1)
y = snow_main_clean['Apr']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=307, test_size=.2)


# %%

# Complex Pipeline

# numeric_features = ['Elev', 'Lat', 'Lon', 'installed', 'Water Year',
#                     'Jan', 'Jan (WE)', 'Feb', 'Feb (WE)', 'Mar', 'Mar (WE)',
#                     'Jan_SA_Avg', 'Feb_SA_Avg', 'Mar_SA_Avg', 'Apr_SA_Avg',
#                     'May_SA_Avg', 'Jun_SA_Avg', 'Jan_WE_Avg', 'Feb_WE_Avg',
#                     'Mar_WE_Avg', 'Apr_WE_Avg', 'May_WE_Avg', 'Jun_WE_Avg']
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
num_transformer = Pipeline([
  ('imputer',SimpleImputer(strategy='median')),
  ('rounder', FunctionTransformer(np.round, validate=False)),
  ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('percentile', SelectPercentile(f_regression, percentile=50)),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features),
    ]
)

pipe_reg = Pipeline(
    steps=[("preprocessor", preprocessor),
    ('model', RandomForestRegressor(random_state=42))]
)



# %%

# Param grid search and running rf model
hyperparameter_grid = {
    'model__n_estimators': [50, 100, 300, 500],
    'model__max_leaf_nodes': [4, 8, 16, 32],
    # 'model__max_features': [3, 5, 10, 15, None],
    # 'model__min_samples_leaf':[1, 3, 5, 10],
    'model__max_depth': [1, 3, 5, 10, 15, 20]
    # 'model__min_samples_split': [2, 5, 10],
    # 'model__min_impurity_decrease': [0.0, 0.1, 0.2],
    # 'model__ccp_alpha': [0.0, 0.1, 0.2]
    }

# GridSearch takes a while to run
# gs = GridSearchCV(pipe_reg, param_grid=hyperparameter_grid, scoring='neg_mean_squared_error', cv=10)
# gs.fit(data_train_X, data_train_y)

rs = RandomizedSearchCV(pipe_reg, param_distributions=hyperparameter_grid, scoring='neg_mean_squared_error', cv=6)
rs.fit(Xtrain, ytrain)

# %%

# View best parameters
rs.best_params_


# %%

y_preds_test = rs.predict(Xtest)


# %%
# Root Mean Squared Error
# np.sqrt(mean_squared_error(ytest, y_preds_test))
np.sqrt(mean_squared_error(ytest, y_preds_test))

# %%
# Compare to ytest variance
np.std(ytest)
# %%

test_comparison = pd.DataFrame({'True Val': ytest, 'Prediction': y_preds_test})

# %%
test_comparison

# %%
ytest
# %%
