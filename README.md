# Stat 486 Snow Accumulation Study

Data: site_snow.csv is a cleaned dataset with observations scraped from the USDA's Air and Water Database: https://wcc.sc.egov.usda.gov/nwcc/snow-course-sites.jsp?state=UT.

Features include the SNOTEL site's name, site elevation (ft), lat, lon, year installed, county, year of observation, and the monthly snow accumulation and respective water equivalence (WE) levels (in).

Null snow accumulation observations were replaced with their respective WE levels multiplied by a "Snow Water Equivalent Coefficient" (average ratio of snow accumulation to WE level across all observations at all months).

## Table of Contents

- [Folders](folders)
  1. [EDA](eda)
     - 
  3. [Models](models)
  4. [Docs](docs)
  5. [fe1_data](fe1_data)
      - Contents: final_data_setup2.py file that setups up the first feature we used and snow_site_new.csv that is a dataset that includes this feature.
  7. [Extra Files](extra-files)
     - Additional EDA: EDA on Feature correlations with FE 1
  8. [Miscellaneous](miscellaneous)

