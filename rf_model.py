#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
X = snow_main_clean.drop(['Apr'], axis=1)
y = snow_main_clean['Apr']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=307, test_size=.2)



# %%

Xtrain

# %%
