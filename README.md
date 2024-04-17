# Stat 486 Snow Accumulation Study

Data: site_snow.csv is a cleaned dataset with observations scraped from the USDA's Air and Water Database: https://wcc.sc.egov.usda.gov/nwcc/snow-course-sites.jsp?state=UT.

Features include the SNOTEL site's name, site elevation (ft), lat, lon, year installed, county, year of observation, and the monthly snow accumulation and respective water equivalence (WE) levels (in).

Null snow accumulation observations were replaced with their respective WE levels multiplied by a "Snow Water Equivalent Coefficient" (average ratio of snow accumulation to WE level across all observations at all months).

## Table of Contents

- [Folders](folders)
  1. [EDA](eda)
     - Contents:
       - eda.py : file that contains the visualizations used on our report
       - regression_analysis.ipynb: file that contains a regression model used for feature significant test
       - streamlit_dashboard.py: dashboard that contains the EDA mentioned in our report
  3. [Models](models)
  4. [Docs](docs)
  5. [Feature Engineering 1](fe1_files)
      - Contents: 
          - final_data_setup2.py: file that sets up the first feature
          - snow_site_new.csv: a dataset that includes this feature.
  7. [Extra Files](extra-files)
     - Contents: 
         - MattJensenCode.ipynb: Matt Jensen's original code that sets up feature two and ran 4 models
         - miscellaneous.py: Aaron Brown's miscellaneous code for his portion.
  8. 

