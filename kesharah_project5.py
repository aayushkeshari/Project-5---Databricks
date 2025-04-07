# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Details:
# MAGIC
# MAGIC Author: Aayush Keshari
# MAGIC
# MAGIC Date: April 6, 2025
# MAGIC
# MAGIC M Number: M15039880
# MAGIC
# MAGIC This project explores the relationship between COVID-19 vaccination rates, case rates, and mortality across different continents during the pandemic. Using data from Our World in Data, the World Bank, and Wikipedia, I will utilize Databricks to prepare, analyze, and visualize data to determine whether higher vaccination rates are associated with reductions in COVID-19 cases and mortality rates globally.
# MAGIC
# MAGIC This notebook is written in Python so the default cell type is Python. However, you can use different languages by using the %LANGUAGE syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/OWID_COVID19_data_4_Project5.csv")
df = spark.read.format("csv") \
.option("header", "true") \
.option("inferSchema", "true") \
.load("dbfs:/FileStore/OWID_COVID19_data_4_Project5.csv")

display(df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import to_date, month, year

# File location and type
file_location = "/FileStore/tables/OWID_COVID19_data_4_Project5.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Load the CSV file into a DataFrame
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Convert the `date` column to DateType if necessary
df = df.withColumn("date", to_date("date", "yyyy-MM-dd"))

# Filter for September and October 2021, exclude null/empty continents, and ensure relevant columns have positive values
df_filtered = df.filter(
    (F.col('continent').isNotNull()) &
    (F.col('continent') != "") &
    (F.col('people_fully_vaccinated_phun') > 0) &
    (F.col('new_cases_pmil') > 0) &
    (year(F.col("date")) == 2021) &
    (month(F.col("date")).isin([9, 10]))
)

# Display the filtered DataFrame to confirm
display(df_filtered)

# COMMAND ----------

# Group by continent and display the row counts for each continent
continent_counts = df_filtered.groupBy("continent").count()
continent_counts.show()

# COMMAND ----------

from pyspark.sql.functions import month, year

# Step 1: Create Month and Year columns
df_filtered = df_filtered.withColumn("month", month("date"))
df_filtered = df_filtered.withColumn("year", year("date"))

# Step 2: Filter for September and October 2021
df_filtered = df_filtered.filter(
    (F.col("year") == 2021) & 
    (F.col("month").isin([9, 10]))
)

# Step 3: Display the total record count
total_count = df_filtered.count()
print(f"Total record count for September and October 2021: {total_count}")

# COMMAND ----------

df_filtered = df.filter(F.col('continent').isNotNull()) \
    .withColumn("year", year("date")) \
    .withColumn("month", month("date")) \
    .filter((F.col('year') == 2021) & (F.col('month').isin([9, 10])))

# Calculate averages by continent and month
# Group by 'continent' and 'month' and calculate averages for the specified metrics
avg_df = df_filtered.groupBy('continent', 'month').agg(
    F.mean('people_fully_vaccinated_phun').alias('average_people_fully_vaccinated'),
    F.mean('new_cases_pmil').alias('average_new_cases'),
    F.mean('excess_mortality').alias('average_excess_mortality')
).orderBy('continent', 'month')

# Display the results
avg_df.show()

# COMMAND ----------

display(avg_df.orderBy('continent', 'month'))

# COMMAND ----------

# Perform correlation analysis
correlation = avg_df.stat.corr('average_people_fully_vaccinated', 'average_new_cases')

# Print the correlation coefficient
print(f"Correlation between average vaccination rate and new cases: {correlation}")
print(f"The correlation coefficient of {correlation:.4f} indicates a moderate positive relationship between vaccination rates and new cases. This suggests that other factors, such as testing rates or high prior infection rates in highly vaccinated regions, might be influencing this unexpected trend.")


# COMMAND ----------

from pyspark.sql import functions as F

# Step 1: Filter rows with missing GDP data
missing_gdp_df = df_filtered.filter(F.col('gdp_per_capita').isNull())
display(missing_gdp_df)

# Step 2: Create a DataFrame for the provided missing GDP data
missing_gdp_data = [
    ("CUB", "North America", "Cuba", 11255.00),
    ("LIE", "Europe", "Liechtenstein", 197505.00),
    ("MCO", "Europe", "Monaco", 240862.00),
    ("SOM", "Africa", "Somalia", 1611.30),
    ("TWN", "Asia", "Taiwan", 67455.00),
    ("SYR", "Asia", "Syria", 2914.50)
]

# Define column names
columns = ["iso_code", "continent", "location", "gdp_per_capita_new"]

# Create the missing GDP DataFrame
missing_gdp_df = spark.createDataFrame(missing_gdp_data, schema=columns)
display(missing_gdp_df)

# Step 3: Merge the missing GDP data into the main DataFrame
df_with_gdp = df_filtered.join(
    missing_gdp_df,
    on=["iso_code", "continent", "location"],
    how="left"
)

# Step 4: Fill missing GDP values with the new column from the joined data
df_with_gdp = df_with_gdp.withColumn(
    "gdp_per_capita",
    F.coalesce(F.col("gdp_per_capita"), F.col("gdp_per_capita_new"))
).drop("gdp_per_capita_new")  # Drop the new column after merging

# Step 5: Display the updated DataFrame to verify missing GDP data is filled
display(df_with_gdp)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import year, month

# Step 1: Filter the Data
# Ensure data is limited to September and October 2021, and rows with valid continent values
filtered_data = df_with_gdp.filter(
    (F.col('continent').isNotNull()) & 
    (F.col('year') == 2021) & 
    (F.col('month').isin([9, 10]))
)

# Step 2: Calculate descriptive statistics
summary_stats = filtered_data.agg(
    F.mean("people_fully_vaccinated_phun").alias("mean_vaccinated"),
    F.stddev("people_fully_vaccinated_phun").alias("stddev_vaccinated"),
    F.count("people_fully_vaccinated_phun").alias("count_vaccinated"),
    
    F.mean("new_cases_pmil").alias("mean_new_cases"),
    F.stddev("new_cases_pmil").alias("stddev_new_cases"),
    F.count("new_cases_pmil").alias("count_new_cases"),
    
    F.mean("excess_mortality").alias("mean_excess_mortality"),
    F.stddev("excess_mortality").alias("stddev_excess_mortality"),
    F.count("excess_mortality").alias("count_excess_mortality"),
    
    F.mean("gdp_per_capita").alias("mean_gdp"),
    F.stddev("gdp_per_capita").alias("stddev_gdp"),
    F.count("gdp_per_capita").alias("count_gdp")
)

# Step 3: Display the summary statistics
display(summary_stats)

# COMMAND ----------

# Correlation between vaccination rates and new cases
correlation_vaccination_cases = avg_df.stat.corr('average_people_fully_vaccinated', 'average_new_cases')

# Correlation between vaccination rates and excess mortality
correlation_vaccination_mortality = avg_df.stat.corr('average_people_fully_vaccinated', 'average_excess_mortality')

# Print correlations
print(f"The correlation between vaccination rates and new cases is {correlation_vaccination_cases:.4f}, indicating a moderate positive relationship. This suggests that regions with higher vaccination rates may also report higher case rates, possibly due to confounding factors like increased testing or high baseline infection rates in these regions.")

print(f"The correlation between vaccination rates and excess mortality is {correlation_vaccination_mortality:.4f}, suggesting a weak negative relationship. This indicates that vaccination may slightly reduce mortality, but the impact is limited by other factors.")

# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert PySpark DataFrame to Pandas for easy plotting
avg_df_pandas = avg_df.toPandas()

# Scatter Plot: Vaccination Rate vs. New Cases Per Million
plt.figure(figsize=(10, 6))
sns.regplot(
    data=avg_df_pandas,
    x='average_people_fully_vaccinated',
    y='average_new_cases',
    line_kws={"color": "red"},
    scatter_kws={"alpha": 0.5}
)
plt.title('Vaccination Rate vs. New Cases Per Million', fontsize=16)
plt.xlabel('Average People Fully Vaccinated (%)', fontsize=12)
plt.ylabel('Average New Cases Per Million', fontsize=12)
plt.grid(True)
plt.show()

# Scatter Plot: Vaccination Rate vs. Excess Mortality
plt.figure(figsize=(10, 6))
sns.regplot(
    data=avg_df_pandas,
    x='average_people_fully_vaccinated',
    y='average_excess_mortality',
    line_kws={"color": "blue"},
    scatter_kws={"alpha": 0.5}
)
plt.title('Vaccination Rate vs. Excess Mortality', fontsize=16)
plt.xlabel('Average People Fully Vaccinated (%)', fontsize=12)
plt.ylabel('Average Excess Mortality', fontsize=12)
plt.grid(True)
plt.show()
