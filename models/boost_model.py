#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, accuracy_score, r2_score
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
from xgboost import XGBClassifier, XGBRFRegressor

# Display all columns
pd.set_option('display.max_columns', None)


#%%
# Import data
# feature engineering: added new features of averages of SA for previous five years (by month)
snow_main_fteng = pd.read_csv('snow_main_new.csv')

#%%
### Drop NAs from original dataset:
# Check for features containing null obs
features_na = snow_main_fteng.columns[snow_main_fteng.isna().any()].tolist()
features_na_drop = features_na[11:17]

# snow_main_clean drops NA's in specified features
snow_main_clean = snow_main_fteng.dropna(subset=features_na_drop) if features_na_drop else snow_main_fteng

# %%

# Remaining Feature Engineering
# Convert object features to factors
snow_main_clean['Site_Name'] = snow_main_clean['Site_Name'].astype('category')
snow_main_clean['County'] = snow_main_clean['County'].astype('category')

# %%
# Parse sets
X = snow_main_clean.drop(['Apr', 'Apr (WE)', 'May', 'May (WE)', 'Jun', 'Jun (WE)'], axis=1)
y = snow_main_clean['Apr']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=307, test_size=.2)



# %%

# Complex Pipeline

numeric_features = ['Elev', 'Lat', 'Lon', 'installed', 'Water Year',
                    'Jan', 'Jan (WE)', 'Feb', 'Feb (WE)', 'Mar', 'Mar (WE)',
                    'Jan_SA_Avg', 'Feb_SA_Avg', 'Mar_SA_Avg', 'Apr_SA_Avg',
                    'May_SA_Avg', 'Jun_SA_Avg', 'Jan_WE_Avg', 'Feb_WE_Avg',
                    'Mar_WE_Avg', 'Apr_WE_Avg', 'May_WE_Avg', 'Jun_WE_Avg']
num_transformer = Pipeline([
  ('imputer',SimpleImputer()),
  ('poly_features', PolynomialFeatures(include_bias=False)),
  # ('fit_transform', BathroomsTextCleaner()),
  ('scaler', StandardScaler()),
  ('percentile', SelectPercentile(f_regression, percentile=20))
])

categorical_features = ['Site_Name', 'County']

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ('percentile', SelectPercentile(f_regression, percentile=50))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features),
    ]
)

pipe_reg = Pipeline(
    steps=[("preprocessor", preprocessor),
    ('model', XGBRFRegressor(n_jobs=-1, random_state=42))]
)



# %%

# Param grid search and running rf model
hyperparameter_grid = {
    # 'model__n_estimators': [50, 100, 500, 900],
    # 'model__learning_rate': [0.05, 0.1, 0.15, 0.20],
    # 'model__min_child_weight': [0.1, 0.5, 1, 2, 10],
    # 'model__max_depth': [2, 3, 5, 10, 15, 20]
    'model__n_estimators': [50, 100, 500, 900],
    'model__learning_rate': [0.05, 0.1, 0.15, 0.20],
    'model__min_child_weight': [0.1, 0.5, 1, 2, 10],
    'model__max_depth': [2, 3, 5, 10, 15, 20],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'model__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'model__reg_lambda': [0, 0.001, 0.005, 0.01, 0.05]
    }


# GridSearch takes a while to run
# gs = GridSearchCV(pipe_reg, param_grid=hyperparameter_grid, scoring='neg_mean_squared_error', cv=10)
# gs.fit(data_train_X, data_train_y)

xg_rs = RandomizedSearchCV(pipe_reg, param_distributions=hyperparameter_grid, scoring='neg_mean_squared_error', cv=6)
xg_rs.fit(Xtrain, ytrain)

# %%

# View best parameters
xg_rs.best_params_


# %%

y_preds_test = xg_rs.predict(Xtest)


# %%
# Root Mean Squared Error
np.sqrt(mean_squared_error(ytest, y_preds_test))

# %%
# Compare to ytest variance
np.std(ytest)
# %%

test_comparison = pd.DataFrame({'True Val': ytest, 'Prediction': y_preds_test})

# %%
test_comparison
# %%
