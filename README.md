# Stat 486 Snow Accumulation Study

For our final project, we conducted an analysis on snow accumulation patterns spanning several decades in Utah.  Our aim was to discern trends in snow accumulation over time and ascertain whether there are significant indications of declining snow levels across the state, and if so, determine an effective predictive model to aid in predicting for the expected snow accumulation totals near the end of the season, which will prove useful for local communities in estimating the environmental conditions for the upcoming year. This repository includes all documents, visualizations, and code files related to this project.

Data: site_snow.csv is a cleaned dataset with observations scraped from the USDA's Air and Water Database: https://wcc.sc.egov.usda.gov/nwcc/snow-course-sites.jsp?state=UT.

Features include the SNOTEL site's name, site elevation (ft), lat, lon, year installed, county, year of observation, and the monthly snow accumulation and respective water equivalence (WE) levels (in).

Null snow accumulation observations were replaced with their respective WE levels multiplied by a "Snow Water Equivalent Coefficient" (average ratio of snow accumulation to WE level across all observations at all months).

## Table of Contents

- Folders
  1. [EDA](eda)
     - Contents:
       - eda.py : file that contains the visualizations used in our report
       - regression_analysis.ipynb: file that contains a regression model used for feature significant test
       - streamlit_dashboard.py: dashboard that contains the EDA mentioned in our report
  3. [Models](models)
      - Contents:
        - final_mod_rf.ipynb: file that contains all KNN, Lasso, SVM, XGBoost, and Random Forest models uses FE 1, FE 2, and a scrapped feature. Also includes SHAP analysis on the best model.
        - nn.ipynb: file that contains a deep learning model using FE 2.
        - rf_fe1_model.ipynb: file that contains a Random Forest model using FE 1.
  5. [Docs](docs)
     - Contents:
       - Final Project Group Names.pdf: pdf containing all group member names 
       - Stat486_Final_Project.pdf: pdf containing the written report
  6. [Feature Engineering 1](fe1_files)
      - Contents: 
          - final_data_setup2.py: file that sets up the first feature
          - snow_site_new.csv: a dataset that includes this feature.
  7. [Extra Files](extra-files)
     - Contents: 
         - MattJensenCode.ipynb: Matt Jensen's original code that sets up feature two and ran 4 models
         - miscellaneous.py: Aaron Brown's miscellaneous code for his portion.
         - xgboost_model.py: Aaron's extra xgboost model.
  8. [Images](images)
     - Contents:
       - Contains all the images used in the report and presentation

