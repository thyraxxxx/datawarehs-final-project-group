"""
Bicycle Theft – Data & EDA
"""

########## Import libraries ##########
import pandas as pd      # table for data
import numpy as np       # numeric helpers
import matplotlib.pyplot as plt  # plotting
import seaborn as sns    
import os                # for file paths

plt.style.use("default")
pd.set_option("display.max_columns", None)

########## Load data ##########

# full path to bicycle theft csv in Downloads\data
fullpath = r"C:\Users\thyra\Downloads\data\bicycle_thefts.csv"

# read csv into df
df = pd.read_csv(fullpath)

print("Data loaded")
print("Shape (rows, columns): ", df.shape)

########## Basic data checks ##########
# table showing first 5 records in bicycle theft records
print(df.head(5))

# array of column names in bicycle theft records
print(df.columns.values)

# table of stats between numeric columns and their measures (mean, std)
print(df.describe())

# table of stats between all columns and their summaries (numeric + categorical)
print(df.describe(include="all"))

# table of column names and their data types
print(df.dtypes)

########## Split numeric and categorical ##########

# list of columns where type is numeric
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# list of columns where type is object (categorical)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns: ", numeric_cols)
print("Categorical columns: ", categorical_cols)

########## Statistical summaries ##########

# series of mean values between numeric columns and their average
print("Means of numeric columns")
print(df[numeric_cols].mean())

# series of median values between numeric columns and their middle value
print("Medians of numeric columns")
print(df[numeric_cols].median())

# table of value counts between each categorical column and its top categories
for col in categorical_cols:
    print("Value counts for", col)
    print(df[col].value_counts(dropna=False).head(10))

########## Missing value analysis ##########
# series of counts between each column and number of missing values
missing_count = df.isnull().sum()

# series of percentages between each column and missing percentage
missing_pct = (missing_count / len(df)) * 100

# table between each column, missing counts, and missing percentages
missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct.round(2)
}).sort_values('missing_pct', ascending=False)

print("Missing values per column")
print(missing_summary)

# bar graph between column names and missing percentage
plt.figure(figsize=(10,6))
missing_summary[missing_summary['missing_count'] > 0]['missing_pct'].plot(kind='bar')
plt.title('Percentage of Missing Values per Column')      # bar graph: column vs missing %
plt.ylabel('Missing %')
plt.tight_layout()
plt.show()

########## Correlation matrix for numeric ##########
if len(numeric_cols) > 1:
    # table of correlations between numeric columns
    corr = df[numeric_cols].corr()
    print("Correlation matrix")
    print(corr)

    # heatmap between numeric columns and their correlation values
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap – Numeric Features')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric columns for correlation")

########## Key visualizations and trends ##########

# columns 
year_col   = "OCC_YEAR"          # year of theft
month_col  = "OCC_MONTH"         # month of theft
dow_col    = "OCC_DOW"           # day of week of theft
hour_col   = "OCC_HOUR"          # hour of day of theft
area_col   = "NEIGHBOURHOOD_158" # neighbourhood / area of theft
status_col = "STATUS"            # status of case (e.g. STOLEN, RECOVERED)
cost_col   = "BIKE_COST"         # reported cost of bike


for col in [year_col, month_col, dow_col, hour_col, area_col, status_col, cost_col]:
    if col not in df.columns:
        print("Column", col, "not found – update this name if needed")

########## Trend: thefts by year ##########
if year_col in df.columns:
    # bar graph between year (x-axis) and number of thefts (y-axis)
    year_counts = df[year_col].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    year_counts.plot(kind='bar')
    plt.title('Trend of Bicycle Thefts by Year')   # shows increase/decrease over years
    plt.xlabel('Year')
    plt.ylabel('Number of Thefts')
    plt.tight_layout()
    plt.show()

########## Trend: thefts by month (seasonality) ##########
if month_col in df.columns:
    # bar graph between month (x-axis) and number of thefts (y-axis)
    month_counts = df[month_col].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    month_counts.plot(kind='bar')
    plt.title('Seasonal Pattern of Bicycle Thefts by Month')  # shows which months are highest
    plt.xlabel('Month')
    plt.ylabel('Number of Thefts')
    plt.tight_layout()
    plt.show()

########## Trend: thefts by day of week ##########
if dow_col in df.columns:
    # bar graph between day of week (x-axis) and number of thefts (y-axis)
    dow_counts = df[dow_col].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    dow_counts.plot(kind='bar')
    plt.title('Bicycle Thefts by Day of Week')  # shows which weekdays/weekend days are higher
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Thefts')
    plt.tight_layout()
    plt.show()

########## Trend: thefts by hour of day ##########
if hour_col in df.columns:
    # bar graph between hour of day (x-axis) and number of thefts (y-axis)
    hour_counts = df[hour_col].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    hour_counts.plot(kind='bar')
    plt.title('Bicycle Thefts by Hour of Day')  # shows peak times 
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Thefts')
    plt.tight_layout()
    plt.show()

########## Distribution: cost of stolen bikes ##########
if cost_col in df.columns:
    # histogram between bike cost (x-axis) and number of bikes (y-axis)
    plt.figure(figsize=(8,4))
    plt.hist(df[cost_col].dropna(), bins=40)
    plt.title('Distribution of Reported Bike Cost')  # shows common price ranges for stolen bikes
    plt.xlabel('Bike Cost ($)')
    plt.ylabel('Number of Bikes')
    plt.tight_layout()
    plt.show()

########## Status of cases (stolen vs recovered) ##########
if status_col in df.columns:
    # bar graph between theft status (x-axis) and number of records (y-axis)
    plt.figure(figsize=(8,4))
    df[status_col].value_counts().plot(kind='bar')
    plt.title('Bicycle Theft Case Status')  # shows how many bikes are recovered vs not
    plt.xlabel('Status')
    plt.ylabel('Number of Records')
    plt.tight_layout()
    plt.show()

########## Hotspot: top neighbourhoods ##########
if area_col in df.columns:
    # bar graph between neighbourhood (x-axis) and theft count (y-axis)
    plt.figure(figsize=(10,4))
    df[area_col].value_counts().head(15).plot(kind='bar')
    plt.title('Top 15 Neighbourhoods for Bicycle Thefts')  # shows hotspot areas in the city
    plt.xlabel('Neighbourhood')
    plt.ylabel('Number of Thefts')
    plt.tight_layout()
    plt.show()

########## Heatmap: day of week vs hour (when thefts happen) ##########
if (dow_col in df.columns) and (hour_col in df.columns):
    # table between day of week and hour of day with theft counts
    pivot_time = pd.crosstab(df[dow_col], df[hour_col])
    # heatmap between (day of week, hour) and theft count
    plt.figure(figsize=(12,5))
    sns.heatmap(pivot_time, cmap='Blues')
    plt.title('Heatmap of Bicycle Thefts: Day of Week vs Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.show()
