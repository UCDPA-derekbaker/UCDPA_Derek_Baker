# Import the packages needed to import and transform the dataframes #
import os
import pandas as pd
import numpy as np
import geopandas as gpd

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

if __name__ == "__main__":
    # please enter your working directory
    main("C:\\Users\\derek.baker\\PycharmProjects\\UCDA2")

print("\nStart importing and transforming CSV and JSON files\n")

# Read in Country Vaccinations CSV and store it as a dataframe called cv_df
cv_df = pd.read_csv("cov19\\country_vaccinations.csv",
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
           axis="columns", inplace=True)
# Check that the columns have been removed
print(cv_df.columns)

# Look to find rows with missing data - count the number of cells with null values
cv_df.isna().sum()

# There are two columns (daily_vaccinations + daily_vaccinations_per_million) both with 187 rows with missing data
# and the other columns don't have any value without this data so I'll remove the rows instead of replacing the values with cv_df.fillna(0)
cv_df = cv_df.dropna()
# Double check for any missing values
cv_df.isna().sum()

# Create a dataframe from the df_cv dataframe to store statistical information which will be used to merge with the GDP by country
cv_df_grp = cv_df.groupby(["country", "iso_code"]).agg(
    {"daily_vaccinations": [np.min, np.max, np.mean, np.median, np.sum, np.count_nonzero]}).reset_index()

# Rename the columns to give the dataframe column names a cleaner look.
cv_df_grp.columns = ["Country", "Country Code", "Daily_Min", "Daily_Max",
                     "Daily_Mean", "Daily_Median", "Daily_Sum", "Days_Administered"]

print("\nFinished importing and transforming Vaccines CSV\n")
############################################################

# Import gdp per capita csv
gdp_df = pd.read_csv("GDP.csv")

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
    (gdp_2018["GDP"] < 2287.8),
    (gdp_2018["GDP"] >= 2287.8) & (gdp_2018["GDP"] < 7655.1),
    (gdp_2018["GDP"] >= 7655.1) & (gdp_2018["GDP"] < 12983.2),
    (gdp_2018["GDP"] >= 12983.2) & (gdp_2018["GDP"] < 19028.9),
    (gdp_2018["GDP"] >= 19028.9)
    ]

# create a list of the values we want to assign for each condition
values = ["Fragile", "Low income", "Lower middle income", "Middle income", "High income"]

# create a new column and use np.select to assign values to it using our lists as arguments
gdp_2018["GDP_Category"] = np.select(conditions, values)

# Check gdp for new column
gdp_2018.head()

# Merge vaccine data with GDP data
cv_gdp = pd.merge(cv_df_grp, gdp_2018, on="Country Code")


print("\nFinished importing and transforming gdp CSV\n")
########################################################

# Find latest population stats to get vaccinations per capita
pop_df = pd.read_csv('WorldPopulation.csv')

# There are no missing values for 2019 so .fillna(method="pad") is not required
# Keep only Country Code and the latest year in the dataset
pop_2019 = pop_df[["Country Code", "2019"]]
# Rename the column "2019" to "Population"
pop_2019 = pop_2019.rename(columns={"2019": "Population"})

# Merge vaccine data with population data
cv_pop = pd.merge(cv_df_grp, pop_2019, on="Country Code")
#  Add a column to get vaccines per capita
cv_pop["Vac_per_Capita"] = cv_pop["Daily_Sum"] / cv_pop["Population"]

print("\n Finished importing and transforming population CSV \n")

# Create a geodataframe of world countries
geo_loc = gpd.read_file("cov19\\world_countries.json")

# merge the data from gdp_2018, cv_df_grp, pop_2019 and geo_loc making the geo_loc the dataset
# to keep all rows from regardless of any data from the other dataframes
cv_gdp_pop_geo = pd.merge(pd.merge(pd.merge(cv_df_grp, gdp_2018, on="Country Code"), pop_2019, on="Country Code"),
                          geo_loc, left_on="Country Code", right_on='id')

# look at the dataframe and make decisions
print(cv_gdp_pop_geo.head())
print(cv_gdp_pop_geo.info())

# Drop duplicate columns
cv_gdp_pop_geo.drop(["Country_x", "Country Code", "Country_y"], axis="columns", inplace=True)

#  Add a column to get vaccines per capita
cv_gdp_pop_geo["Vac_per_Capita"] = cv_gdp_pop_geo["Daily_Sum"] / cv_gdp_pop_geo["Population"]



# Convert cv_gdp_pop_geo to a Geodataframe
geo_cv_gdp_pop_loc = gpd.GeoDataFrame(cv_gdp_pop_geo, crs="EPSG:4326", geometry=cv_gdp_pop_geo["geometry"])

# Merge vaccine data with location data
cv_loc = pd.merge(cv_df_grp, geo_loc, left_on="Country Code", right_on="id")
# Convert cv_loc to a Geodataframe
geo_cv_loc = gpd.GeoDataFrame(cv_loc, crs="EPSG:4326", geometry=cv_loc["geometry"])

# Create Area column #
# To get correct calculations, convert the crs
geo_cv_gdp_pop_loc_3857 = geo_cv_gdp_pop_loc.to_crs(epsg=3857)
# Add Area column to geo_cv_gdp_pop_loc as kilometers squared
geo_cv_gdp_pop_loc["Area"] = geo_cv_gdp_pop_loc_3857.geometry.area / 10**6

# Create a Density column to show countries densely populated per squared kilometer
geo_cv_gdp_pop_loc["Density"] = geo_cv_gdp_pop_loc.Population / geo_cv_gdp_pop_loc.Area

# Create a geometry center point for each country
geo_cv_gdp_pop_loc["centre"] = geo_cv_gdp_pop_loc.geometry.centroid
# The Bokeh visual won't allow me to to use the geo_cv_gdp_pop_loc dataframe, "Point is not JSON serializable"
# so I'll drop "centre" from the dataframe
cv_gdp_pop_loc = geo_cv_gdp_pop_loc
cv_gdp_pop_loc.drop(["centre", "geometry"], axis="columns", inplace=True)

print("\n Datasets added transformed and merged with no problems \n")

######################################################################


###########################################
# Explore the data #
###########################################

# Find out how many vaccines are given daily to all countries
total_daily_vac = cv_df.groupby("date")["daily_vaccinations"].sum()
total_daily_vac = pd.DataFrame(total_daily_vac)
print(total_daily_vac)

# Check how many counties had recorded vaccinations
print(len(cv_df_grp))


## Create risk factor analysis
# Risk factor
cv_gdp_pop_geo["Risk_Factor"] = cv_gdp_pop_geo["Population"] * cv_gdp_pop_geo["Density"] / 1000000000
# Current risk factor - Going with the assumption that two vaccine doses are needed, I divide vac_per_capita by 2
cv_gdp_pop_geo["Current_Risk_Factor"] = cv_gdp_pop_geo["Risk_Factor"] * (1 - (cv_gdp_pop_geo["Vac_per_Capita"]/2))
# Risk reduced by
cv_gdp_pop_geo["Risk_Factor_Reduced"] = 1 - cv_gdp_pop_geo["Current_Risk_Factor"] / cv_gdp_pop_geo["Risk_Factor"]

# Add the risk analysis to cv_gdp_pop_loc for bokeh analysis
cv_gdp_pop_loc["Risk_Factor"] = cv_gdp_pop_loc["Population"] * cv_gdp_pop_loc["Density"] / 1000000000
# Current risk factor - Going with the assumption that two vaccine doses are needed, I divide vac_per_capita by 2
cv_gdp_pop_loc["Current_Risk_Factor"] = cv_gdp_pop_loc["Risk_Factor"] * (1 - (cv_gdp_pop_loc["Vac_per_Capita"]/2))
# Risk reduced by
cv_gdp_pop_loc["Risk_Factor_Reduced"] = 1 - cv_gdp_pop_loc["Current_Risk_Factor"] / cv_gdp_pop_loc["Risk_Factor"]
###########################################
#  Visualisations #
###########################################

# Import the tools need to do the visualisations #
import seaborn as sns
import matplotlib.pyplot as plt

# Set a GDP category order list for use in visualisations
GDP_Cat_order=["Fragile", "Low income", "Lower middle income", "Middle income", "High income"]

###########################################
## Fig. 1 ## Matplotlib and seaborn
###########################################

# Simple Seaborn Count plot showing days administered by GDP category
plt.subplot(2, 2, 1)
sns.countplot(x="GDP_Category", order=GDP_Cat_order, data=cv_gdp)
#plt.xticks(rotation=10)
plt.title("Count of Countries by GDP category")
plt.xlabel("GDP Category")
plt.ylabel("Count of Countries")
# Activate the middle subplot
plt.subplot(2, 2, 2)
sns.barplot(x="GDP_Category", y="Days_Administered", data=cv_gdp, order=GDP_Cat_order)
plt.title('Days Administered Regression on GDP')
plt.xlabel('GDP')
plt.ylabel("Days Administered")
# Activate the bottom subplot
plt.subplot(2, 2, 3)
sns.regplot(x="GDP", y="Days_Administered", data=cv_gdp, color="#FFB000")
plt.title('Days Administered Regression on GDP')
plt.xlabel('GDP')
plt.ylabel("Days Administered")
# Polynomial regression
plt.subplot(2, 2, 4)
sns.regplot(data=cv_gdp_pop_geo, x="GDP", y="Vac_per_Capita", order=2)
plt.title("Polynomial regression vac per capita on GDP")
plt.xlabel("GDP")
plt.ylabel("Vac per Capita")
plt.show()

###########################################
## Fig. 2 ## bokeh
###########################################

# Bokeh - try to add Select menu into this model p280+
# Import bokeh tools #
from bokeh.plotting import figure
from bokeh.io import output_file, show, curdoc, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row, column, widgetbox
from bokeh.models import CategoricalColorMapper, Slider, Select, HoverTool

# Create figure one with x axis type set as datetime #
p1 = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='Vaccinations',
            tools=["box_select", "crosshair", "pan", "wheel_zoom"])

# Plot date along the x axis and vaccinations along the y axis
p1.line(total_daily_vac.index, total_daily_vac["daily_vaccinations"])
hover1 = HoverTool(tooltips=None, mode="hline")
p1.add_tools(hover1)

# Create the second figure #
# Convert cv_gdp_pop_loc to a ColumnDataSource: source
source = ColumnDataSource(cv_gdp_pop_loc)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(
    factors=["Fragile", "Low income", "Lower middle income", "Middle income", "High income"],
    palette=["red", "orange", "blue", "green", "yellow"])

p2 = figure(tools=["box_select", "pan", "wheel_zoom"], x_axis_label='GDP', y_axis_label='Current Risk Factor')
# Plot vaccinations along the x axis and density along the y axis
p2.circle("GDP", "Current_Risk_Factor", selection_color="black", source=source,
          nonselection_alpha=0.1, color=dict(field='GDP_Category', transform=color_mapper))
# Create and add the hover tool for the second visual
hover2 = HoverTool(tooltips=[("name", "@name"),
                             ("GDP_Category", "@GDP_Category"),
                             ("Risk_Factor", "@Risk_Factor"),
                             ("Current_Risk_Factor", "@Current_Risk_Factor"),
                             ("Risk_Factor_Reduced", "@Risk_Factor_Reduced")])
p2.add_tools(hover2)

# Create the third figure #
p3 = figure(tools=["box_select", "pan", "wheel_zoom"], x_axis_label='GDP', y_axis_label='Risk Factor Reduced')
# Plot date along the x axis and vaccinations along the y axis
p3.circle("GDP", "Risk_Factor_Reduced",  selection_color="black", source=source,
          nonselection_alpha=0.1, color=dict(field='GDP_Category', transform=color_mapper),
          legend_group="GDP_Category")
