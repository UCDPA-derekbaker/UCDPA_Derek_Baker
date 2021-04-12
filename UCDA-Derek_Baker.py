# Import the packages needed to run the code in this file
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

# Read in Country Vaccinations CSV and store it as a dataframe called cv_df
cv_df = pd.read_csv('C:\\Users\\derek.baker\\PycharmProjects\\UCDA2\\cov19\\country_vaccinations.csv', parse_dates=["date"], index_col=["date"]).sort_index()

# Inspect the dataframe
print(cv_df.shape)  # Find the number of columns and rows
print(cv_df.index)  # The index here is now DatetimeIndex
print(cv_df.columns)  # A clean look at column names
print(cv_df.head())  # A look at the first few rows of the dataset
print(cv_df.info())  # A look the information of each column i.e. data type and missing values
print(cv_df.describe())  # Get summary stats for each column
print(type(cv_df)) # This is a pandas dataframe
print(cv_df.values)  # Prints a 2 dimensional Numpy array of values

# Delete columns not needed for this analysis
# If all the columns to delete were the last columns, I would have used a slice command cv_df = cv_df.loc[:,:"vaccines"]
cv_df.drop(["total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "total_vaccinations_per_hundred",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred","daily_vaccinations_raw", "source_name", "source_website"],
           axis='columns', inplace=True)
# Check that the columns have been removed
print(cv_df.columns)

# Look to find rows with missing data - count the number of cells with null values
cv_df.isna().sum()

# There are two columns (daily_vaccinations + daily_vaccinations_per_million) both with 187 rows with missing data
# and the other columns don't have any value without this data so I'll remove the rows instead of replacing the values with cv_df.fillna(0)
cv_df = cv_df.dropna()
# Double check for any missing values
cv_df.isna().sum()

# Find out how many vaccines are given daily to all countries
total_daily_vac = cv_df.groupby("date")["daily_vaccinations"].sum()
print(total_daily_vac) # Data is accurate

# Find out how many days each country has received vaccinations
cv_df_group_daily_vacs = cv_df[['country', 'daily_vaccinations']].groupby("country")
cv_df_daily_vacs_by_country = total_daily_vac.agg('count').sort_values("daily_vaccinations", ascending=False)
print(cv_df_daily_vacs_by_country)

# Create a dataframe from the df_cv dataframe to store statistical information which will be used to merge with the GDP by country
cv_df_grp = cv_df.groupby(["country","iso_code"]).agg({'daily_vaccinations': [np.min, np.max, np.mean, np.median, np.sum]}).reset_index()


# Now pivot the dataset and reduce the countries to match the cv_df dataset
# Even though the  inner join would reduce the columns returned for gdp_df any, if this was a large dataset,
# if would improve the performance going forward
gdp_df_reduced = gdp_df[gdp_df["Country"].isin(cv_countries)]

# Pivot the dataframe using .melt and replace missing values with the pad method so that the latest year of data
# will be populated for all countries
gdp_pivot = gdp_df_reduced.melt(id_vars=["Country","Country Code"], value_name="gdp").fillna(method="pad")