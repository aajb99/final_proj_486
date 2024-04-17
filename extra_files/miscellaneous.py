###################################################
### Final setup original code #####################
###################################################

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

snow_main = pd.read_csv('../site_snow.csv')

#%%
len(snow_main['County'].unique())

#%%

snow_main['Water Year'].min()


# %%

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

#%%

# Add averages
for index, row in snow_main.iterrows():

    # Set up iteration vars
    site_name_i = snow_main['Site_Name'][index]
    current_year = snow_main['Water Year']
    start_year = current_year - 5
    jan_five_yr_avg = 0
    feb_five_yr_avg = 0
    mar_five_yr_avg = 0
    apr_five_yr_avg = 0
    may_five_yr_avg = 0
    jun_five_yr_avg = 0

    df_current_site = snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] >= start_year)]

    # if snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == start_year)].shape[0] > 0:
    # if snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'])]:
    # if df_current_site[df_current_site['Water Year'] == first_yr]['Jan'].values[0]:
    if df_current_site['Water Year'].nunique() >= 5:

        # 5 prior years to include:
        first_yr = snow_main['Water Year'][index] - 5
        second_yr = snow_main['Water Year'][index] - 4
        third_yr = snow_main['Water Year'][index] - 3
        fourth_yr = snow_main['Water Year'][index] - 2
        fifth_yr = snow_main['Water Year'][index] - 1

        if df_current_site[df_current_site['Water Year'] == first_yr]['Jan'].empty:
            print('failure')
            print(index)

        else:

            # jan_five_yr_avg = (((df_current_site[df_current_site['Water Year'] == first_yr]['Jan'].iloc[0]) + (df_current_site[df_current_site['Water Year'] == second_yr]['Jan'].iloc[0]) + \
            #     (df_current_site[df_current_site['Water Year'] == third_yr]['Jan'].iloc[0]) + (df_current_site[df_current_site['Water Year'] == fourth_yr]['Jan'].iloc[0]) + \
            #         (df_current_site[df_current_site['Water Year'] == fifth_yr]['Jan'].iloc[0])) / 5)

            jan_five_yr_avg = df_current_site[df_current_site['Water Year'].isin([first_yr, second_yr, third_yr, fourth_yr, fifth_yr])]['Jan'].mean()
            snow_main['Jan_SA_Avg'][index] = jan_five_yr_avg

        # feb_five_yr_avg = (df_current_site[df_current_site['Water Year'] == first_yr]['Feb'] + df_current_site[df_current_site['Water Year'] == second_yr]['Feb'] + \
        #                    df_current_site[df_current_site['Water Year'] == third_yr]['Feb'] + df_current_site[df_current_site['Water Year'] == fourth_yr]['Feb'] + \
        #                     df_current_site[df_current_site['Water Year'] == fifth_yr]['Feb']) / 5
        
        # mar_five_yr_avg = (df_current_site[df_current_site['Water Year'] == first_yr]['Mar'] + df_current_site[df_current_site['Water Year'] == second_yr]['Mar'] + \
        #                    df_current_site[df_current_site['Water Year'] == third_yr]['Mar'] + df_current_site[df_current_site['Water Year'] == fourth_yr]['Mar'] + \
        #                     df_current_site[df_current_site['Water Year'] == fifth_yr]['Mar']) / 5
        
        # apr_five_yr_avg = (df_current_site[df_current_site['Water Year'] == first_yr]['Apr'] + df_current_site[df_current_site['Water Year'] == second_yr]['Apr'] + \
        #                    df_current_site[df_current_site['Water Year'] == third_yr]['Apr'] + df_current_site[df_current_site['Water Year'] == fourth_yr]['Apr'] + \
        #                     df_current_site[df_current_site['Water Year'] == fifth_yr]['Apr']) / 5
        
        # may_five_yr_avg = (df_current_site[df_current_site['Water Year'] == first_yr]['May'] + df_current_site[df_current_site['Water Year'] == second_yr]['May'] + \
        #                    df_current_site[df_current_site['Water Year'] == third_yr]['May'] + df_current_site[df_current_site['Water Year'] == fourth_yr]['May'] + \
        #                     df_current_site[df_current_site['Water Year'] == fifth_yr]['May']) / 5

        # jun_five_yr_avg = (df_current_site[df_current_site['Water Year'] == first_yr]['Jun'] + df_current_site[df_current_site['Water Year'] == second_yr]['Jun'] + \
        #                    df_current_site[df_current_site['Water Year'] == third_yr]['Jun'] + df_current_site[df_current_site['Water Year'] == fourth_yr]['Jun'] + \
        #                     df_current_site[df_current_site['Water Year'] == fifth_yr]['Jun']) / 5

        # # Jan
        # snow_main['Jan_SA_Avg'][index] = jan_five_yr_avg
        # # Feb
        # snow_main['Feb_SA_Avg'][index] = feb_five_yr_avg
        # # Mar
        # snow_main['Mar_SA_Avg'][index] = mar_five_yr_avg
        # # Apr
        # snow_main['Apr_SA_Avg'][index] = apr_five_yr_avg
        # # May
        # snow_main['May_SA_Avg'][index] = may_five_yr_avg
        # # Jun
        # snow_main['Jun_SA_Avg'][index] = jun_five_yr_avg
        
    else:

        print('failure')

        # Jan
        snow_main['Jan_SA_Avg'][index] = None
        # Feb
        snow_main['Feb_SA_Avg'][index] = None
        # Mar
        snow_main['Mar_SA_Avg'][index] = None
        # Apr
        snow_main['Apr_SA_Avg'][index] = None
        # May
        snow_main['May_SA_Avg'][index] = None
        # Jun
        snow_main['Jun_SA_Avg'][index] = None

#%%
        
snow_main.iloc[1060:1070]


# %%
    
# snow_main.iloc[1060:1070]
        
snow_main[snow_main['Site_Name'] == 'Dills Camp']['Water Year'].nunique()


# %%
# test

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

# Add averages
for index, row in snow_main.iterrows():

    # Set up iteration vars
    site_name_i = row['Site_Name']
    current_year = row['Water Year']
    start_year = current_year - 5
    jan_five_yr_avg = 0
    feb_five_yr_avg = 0
    mar_five_yr_avg = 0
    apr_five_yr_avg = 0
    may_five_yr_avg = 0
    jun_five_yr_avg = 0

    if snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == start_year)].shape[0] > 0:
        
        # 5 prior years to include:
        first_yr = current_year - 5
        second_yr = current_year - 4
        third_yr = current_year - 3
        fourth_yr = current_year - 2
        fifth_yr = current_year - 1
        
        jan_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['Jan'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['Jan'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['Jan'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['Jan'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['Jan'].values[0]) / 5

        feb_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['Feb'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['Feb'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['Feb'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['Feb'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['Feb'].values[0]) / 5
        
        mar_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['Mar'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['Mar'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['Mar'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['Mar'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['Mar'].values[0]) / 5
        
        apr_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['Apr'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['Apr'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['Apr'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['Apr'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['Apr'].values[0]) / 5
        
        may_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['May'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['May'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['May'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['May'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['May'].values[0]) / 5
        
        jun_five_yr_avg = (snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == first_yr)]['Jun'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == second_yr)]['Jun'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == third_yr)]['Jun'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fourth_yr)]['Jun'].values[0] + \
                           snow_main[(snow_main['Site_Name'] == site_name_i) & (snow_main['Water Year'] == fifth_yr)]['Jun'].values[0]) / 5
        
    # Jan
    snow_main['Jan_SA_Avg'][index] = jan_five_yr_avg
    # Feb
    snow_main['Feb_SA_Avg'][index] = feb_five_yr_avg
    # Mar
    snow_main['Mar_SA_Avg'][index] = mar_five_yr_avg
    # Apr
    snow_main['Apr_SA_Avg'][index] = apr_five_yr_avg
    # May
    snow_main['May_SA_Avg'][index] = may_five_yr_avg
    # Jun
    snow_main['Jun_SA_Avg'][index] = jun_five_yr_avg







###################################################
### Final setup 2 code ############################
###################################################
    
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





###################################################
### RF model code #################################
###################################################

# Checking years
# snow_main_clean['Water Year'].unique().max()