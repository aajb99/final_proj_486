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

snow_main = pd.read_csv('site_snow.csv')
snow_main['Jan'] = snow_main['Jan'].round(0)


#%%

###########################
### Feature Engineering ###
###########################

# Prior 5 years monthly snow accumulation averages

# Jan
snow_main['Jan_SA_Avg'] = None
# Feb
snow_main['Feb_SA_Avg'] = None
# Mar
snow_main['Mar_SA_Avg'] = None
# Apr
snow_main['Apr_SA_Avg'] = None
# May
snow_main['May_SA_Avg'] = None
# Jun
snow_main['Jun_SA_Avg'] = None

# %%

# # snow_main_new = pd.DataFrame(columns = snow_main.columns)
# snow_main_jan = []

# # Add averages
# for location in snow_main['Site_Name']:

#     df_loc_small = snow_main[snow_main['Site_Name'] == location]

#     # Set up iteration vars
#     # current_year = snow_main['Water Year']
#     # start_year = current_year - 5

#     for index, row in df_loc_small.iterrows():

#         jan_five_yr_avg = 0
#         feb_five_yr_avg = 0
#         mar_five_yr_avg = 0
#         apr_five_yr_avg = 0
#         may_five_yr_avg = 0
#         jun_five_yr_avg = 0

#         # if df_loc_small['Water Year'].nunique() >= 5:
#         if index >= 5:
#             print(df_loc_small['Site_Name'])
#             # jan_five_yr_avg = df_loc_small['Jan'].rolling(6, closed = 'left', min_periods = 4).mean().iloc[-1]
#             jan_five_yr_avg = snow_main[snow_main['Site_Name'] == 'Agua Canyon']['Jan'].iloc[(index - 5):index].mean()
#             row['Jan_SA_Avg'] = jan_five_yr_avg
#             snow_main_jan = snow_main_jan.append(row, ignore_index = True)

#         else:
#             print('Failed to compute avg.')
#             row['Jan_SA_Avg'] = None
#             snow_main_jan = snow_main_jan.append(row, ignore_index = True)


# snow_main_new = pd.DataFrame(snow_main_jan)
# snow_main_new.to_csv('snow_main_new.csv', index=False)





# Initialize an empty list to store rows
snow_main_new_rows = []

# Add averages
for location in snow_main['Site_Name']:
    df_loc_small = snow_main[snow_main['Site_Name'] == location].reset_index(drop=True)
    # print(df_loc_small['Site_Name'])

    # Set up iteration vars
    for index, row in df_loc_small.iterrows():
        # print(row['Jan'])
        jan_five_yr_avg = 0
        feb_five_yr_avg = 0
        mar_five_yr_avg = 0
        apr_five_yr_avg = 0
        may_five_yr_avg = 0
        jun_five_yr_avg = 0
        
        # Check if there are at least 5 rows before the current row
        if index >= 5:
            # jan_five_yr_avg = df_loc_small[(df_loc_small['Site_Name'] == location)]['Jan'].iloc[(index - 5):index].mean()
            jan_five_yr_avg = df_loc_small['Jan'].iloc[(index - 5):index].mean()
            row['Jan_SA_Avg'] = jan_five_yr_avg
            feb_five_yr_avg = df_loc_small['Feb'].iloc[(index - 5):index].mean()
            row['Feb_SA_Avg'] = feb_five_yr_avg
            mar_five_yr_avg = df_loc_small['Mar'].iloc[(index - 5):index].mean()
            row['Mar_SA_Avg'] = mar_five_yr_avg
            apr_five_yr_avg = df_loc_small['Apr'].iloc[(index - 5):index].mean()
            row['Apr_SA_Avg'] = apr_five_yr_avg
            may_five_yr_avg = df_loc_small['May'].iloc[(index - 5):index].mean()
            row['May_SA_Avg'] = may_five_yr_avg
            jun_five_yr_avg = df_loc_small['Jun'].iloc[(index - 5):index].mean()
            row['Jun_SA_Avg'] = jun_five_yr_avg
            print(index)

        else:
            print('Failed to compute avg.')
            row['Jan_SA_Avg'] = None
        
        # Append the modified row to the list
        snow_main_new_rows.append(row)

# Convert the list of rows to a DataFrame
snow_main_new = pd.DataFrame(snow_main_new_rows)

# Save the DataFrame to CSV
snow_main_new.to_csv('snow_main_new.csv', index=False)

print("DataFrame saved to CSV successfully.")




# %%

# snow_main[snow_main['Site_Name'] == 'Agua Canyon']['Jan'].iloc[0:2].mean()


# # %%
# snow_main[snow_main['Site_Name'] == 'Agua Canyon']['Jan']

# # %%
# snow_main_new
# # %%
# snow_main_new = pd.DataFrame(columns = snow_main.columns)

# df_loc_small = snow_main[snow_main['Site_Name'] == 'Agua Canyon']

# for index, row in df_loc_small.iterrows():

#         jan_five_yr_avg = 0
#         feb_five_yr_avg = 0
#         mar_five_yr_avg = 0
#         apr_five_yr_avg = 0
#         may_five_yr_avg = 0
#         jun_five_yr_avg = 0

#         if index >= 5:

#             # jan_five_yr_avg = df_loc_small['Jan'].rolling(6, closed = 'left', min_periods = 4).mean().iloc[-1]
#             jan_five_yr_avg = snow_main[snow_main['Site_Name'] == 'Agua Canyon']['Jan'].iloc[(index - 5):index].mean()
#             # row['Jan_SA_Avg'] = jan_five_yr_avg
#             print(jan_five_yr_avg)
#             # print(row)
#             snow_main_new = snow_main_new.append(row, ignore_index = True)


#         else:
#             print('Failed to compute avg.')
#             # row['Jan_SA_Avg'] = None
#             # snow_main_new = snow_main_new.append(row, ignore_index = True)

# # %%

# 



# %%
