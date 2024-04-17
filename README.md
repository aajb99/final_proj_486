# Stat 486 Snow Accumulation Study

Data: site_snow.csv is a cleaned dataset with observations scraped from the USDA's Air and Water Database: https://wcc.sc.egov.usda.gov/nwcc/snow-course-sites.jsp?state=UT.

Features include the SNOTEL site's name, site elevation (ft), lat, lon, year installed, county, year of observation, and the monthly snow accumulation and respective water equivalence (WE) levels (in).

Null snow accumulation observations were replaced with their respective WE levels multiplied by a "Snow Water Equivalent Coefficient" (average ratio of snow accumulation to WE level across all observations at all months).

## Table of Contents

- [Folders](#folders)
  1. [EDA](#eda)
  2. [Models](#models)
  3. [Docs](#docs)
  4. [Extra Files](#extra-files)
     - [Additional EDA: EDA on Feature correlations with FE 1](#additional-eda-eda-on-feature-correlations-with-fe-1)
  5. [Miscellaneous](#miscellaneous)

## Folders
1. **EDA**
2. **Models**
3. **Docs**
4. **Extra Files**
   - Additional EDA: EDA on Feature correlations with FE 1
5. **Miscellaneous**
