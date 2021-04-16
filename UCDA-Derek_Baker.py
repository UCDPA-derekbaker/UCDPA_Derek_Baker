## Import the packages needed to run the code in this file
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os

# set working directory incorrectly to test the function setting the correct working directory
os.chdir("C:\\Users\\derek.baker\\PycharmProjects")

# Create function to set the correct working directory
def main(enter_path):
    print("Current Working Directory ", os.getcwd())
    try:
        if os.getcwd() == enter_path:
            print("Directory is already set to that path")
        # Change the current working Directory
        else:
            os.chdir(enter_path)
            print("Directory changed")
    except OSError:
        print("Can't change the Current Working Directory")
    print("Current Working Directory ", os.getcwd())

if __name__ == '__main__':
    # please enter your working directory
    main("C:\\Users\\derek.baker\\PycharmProjects\\UCDA2")

print("\nStart importing and transforming CSV and JSON files\n")

# Read in Country Vaccinations CSV and store it as a dataframe called cv_df
cv_df = pd.read_csv('cov19\\country_vaccinations.csv',
                    parse_dates=["date"], index_col=["date"]).sort_index()

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
            "people_fully_vaccinated_per_hundred", "daily_vaccinations_raw", "source_name", "source_website"],
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
print(total_daily_vac)

# Find out how many days each country has received vaccinations
cv_df_daily_vacs_by_country = cv_df.groupby('country').count()['daily_vaccinations']
print(cv_df_daily_vacs_by_country)

# Create a dataframe from the df_cv dataframe to store statistical information which will be used to merge with the GDP by country
cv_df_grp = cv_df.groupby(["country", "iso_code"]).agg(
    {'daily_vaccinations': [np.min, np.max, np.mean, np.median, np.sum, np.count_nonzero]}).reset_index()
# Give the columns proper names
cv_df_grp.columns = ["Country", "Country Code", "Daily_Min", "Daily_Max",
                     "Daily_Mean", "Daily_Median", "Daily_Sum", "Days_Administered"]

print("\nFinished importing and transforming Vaccines CSV\n")

# Import gdp per capita csv
gdp_df = pd.read_csv('GDP.csv')

# Pivot the dataframe using .melt and replace missing values with the pad method so that the latest year of data
# will be populated for all countries and set the column names for the pivoted columns
gdp_pivot = gdp_df.melt(id_vars=["Country", "Country Code"],
                                var_name="Year", value_name="GDP").fillna(method="pad")

# Create a gdp_2018 dataframe to contain the data from gdp_pivot with just values for the year 2018
gdp_2018 = gdp_pivot[gdp_pivot["Year"] == "2018"]

# Create a gdp_levels for categorisation purposes
gdp_levels = gdp_2018[gdp_2018["Country"].isin(
    ["Low income", "Lower middle income", "Middle income", "Upper middle income ", "High income"])]

# Create a column in gdp_2018 with categorisation set by gdp_levels
# create a list of our conditions values set by looking at the values in gdp_levels - will try to automate this
conditions = [
    (gdp_2018['GDP'] < 2287.8),
    (gdp_2018['GDP'] >= 2287.8) & (gdp_2018['GDP'] < 7655.1),
    (gdp_2018['GDP'] >= 7655.1) & (gdp_2018['GDP'] < 12983.2),
    (gdp_2018['GDP'] >= 12983.2) & (gdp_2018['GDP'] < 19028.9),
    (gdp_2018['GDP'] >= 19028.9)
    ]

# create a list of the values we want to assign for each condition
values = ["Lowest income", "Low income", "Lower middle income", "Middle income", "High income"]

# create a new column and use np.select to assign values to it using our lists as arguments
gdp_2018['GDP Category'] = np.select(conditions, values)

# Check gdp for new column
gdp_2018.head()


print("\nFinished importing and transforming gdp CSV\n")

# Find latest population stats to get vaccinations per capita
pop_df = pd.read_csv('WorldPopulation.csv')

# There are no missing values for 2019 so .fillna(method="pad") is not required
# Keep only Country Code and the latest year in the dataset
pop_2019 = pop_df[["Country Code", "2019"]]
# Rename the column "2019" to "Population"
pop_2019 = pop_2019.rename(columns={'2019': 'Population'})

print("\n Finished importing and transforming population CSV \n")

# Create a geodataframe of world countries
geo_loc = gpd.read_file('cov19\\world_countries.json')

# merge the data from gdp_2018, cv_df_grp, pop_2019 and geo_loc making the geo_loc the dataset to keep all rows from regardless of any data from the other dataframes
cv_gdp_pop_geo = pd.merge(pd.merge(pd.merge(cv_df_grp, gdp_2018, on='Country Code'), pop_2019, on='Country Code'),
                          geo_loc, left_on='Country Code', right_on='id', how='right')

# look at the geodataframe and make decisions
print(cv_gdp_pop_geo.head())
print(cv_gdp_pop_geo.info())

# Drop duplicate columns
cv_gdp_pop_geo.drop(["Country_x", "Country Code", "Country_y"], axis='columns', inplace=True)


#  Add a column to get vaccines per capita
cv_gdp_pop_geo["Vac_per_Capita"] = cv_gdp_pop_geo["Daily_Sum"] / cv_gdp_pop_geo["Population"]


print("\n Datasets added transformed and merged with no problems \n")
