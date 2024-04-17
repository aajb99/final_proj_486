#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

#%%

snow_main = pd.read_csv('../fe1_data/snow_main_new.csv')


# %%

snow_main.describe()


#%%

### Drop NAs from original dataset:
# Check for features containing null obs
features_na = snow_main.columns[snow_main.isna().any()].tolist()
features_na_drop = features_na[11:17]

# snow_main_clean drops NA's in specified features
snow_main_clean = snow_main.dropna(subset=features_na_drop) if features_na_drop else snow_main

#%%
# Convert object features to factors
snow_main_clean['Site_Name'] = snow_main_clean['Site_Name'].astype('category')
snow_main_clean['County'] = snow_main_clean['County'].astype('category')

#%%
# Drop unused features in prediction
snow_main_clean = snow_main_clean.drop(['Apr (WE)', 'May', 'May (WE)', 'Jun', 'Jun (WE)'], axis=1)


#%%

# Heatmaps #
############

snow_clean_numeric = snow_main_clean.select_dtypes(include=['int64', 'float64'])
snow_clean_num_mat = snow_clean_numeric.corr()

sns.heatmap(snow_clean_num_mat, annot=False, cmap='YlGnBu')
plt.title('Correlation Heatmap')
# plt.xticks(rotation=1, fontsize=8)


# %%

sns.pairplot(snow_main_clean, hue='Site_Name', palette='Set2')
plt.suptitle('Pair Plot of Cleaned Snow Data')


# %%
pplot1_data = snow_main_clean.iloc[:, [0,1,4,6,7,8,9,10,11,12,13]]
# %%
sns.pairplot(pplot1_data, hue='Site_Name', palette='Set2')
plt.suptitle('Pair Plot of Cleaned Snow Data')
# %%
