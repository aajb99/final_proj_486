Data: site_snow.csv is a cleaned dataset with observations scraped from the USDA's Air and Water Database: https://wcc.sc.egov.usda.gov/nwcc/snow-course-sites.jsp?state=UT. 
Features include the SNOTEL site's name, site elevation (ft), lat, lon, year installed, county, year of observation, and the monthly snow accumulation and respective water equivalence (WE) levels (in).
Null snow accumulation observations were replaced with their respective WE levels multiplied by a "Snow Water Equivalent Coefficient" (average ratio of snow accumulation to WE level across all observations 
at all months).
