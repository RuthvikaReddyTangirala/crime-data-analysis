# ## Exploratory Data Analysis (EDA):

# Overall Crime Trends:
# - Calculate and plot the total number of crimes per year to visualize the trends.

#importing all the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime
from datetime import timezone
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("data\data_preprocessing.csv")
#Converting the object datatype of Date Rptd, DATE OCC into date time object
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Custom function to pad single-digit hours and minutes
#Converting the ineteger datatype into time object
def pad_time(time_str):
    time_str = str(time_str).strip()
    if ':' in time_str:  # Check if the format includes a colon
        try:
            # Try to parse the time using a standard format
            return pd.to_datetime(time_str, format='%H:%M:%S').time()
        except ValueError:
            return pd.NaT  # Return Not-a-Time for invalid formats
    else:
        # Process time in 'HHMM' format
        time_str = time_str.zfill(4)  # Pad with leading zeros
        hours = int(time_str[:2])
        minutes = int(time_str[2:])
        if hours > 23 or minutes > 59:
            return pd.NaT  # Invalid time, return Not-a-Time
        return pd.to_datetime(f"{hours:02}:{minutes:02}", format='%H:%M').time()

# Apply the custom function and convert to datetime
df['TIME OCC'] = df['TIME OCC'].apply(pad_time)

#4.1 Overall crime trends
df = df[df['Date Rptd'].dt.year >= 2020]
crime_trends = df['Date Rptd'].groupby(df['Date Rptd'].dt.to_period('Y')).agg('count')
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=crime_trends.index.astype(str), y=crime_trends.values, palette='viridis')
plt.title('Overall Crime Trends from 2020 to Present Year')
plt.xlabel('Year')
plt.ylabel('Number of Reported Crimes')

for index, value in enumerate(crime_trends):
    plt.text(index, value + 1000, str(value), ha='center', va='bottom', color="blue")
max_year = crime_trends.idxmax().strftime('%Y')
min_year = crime_trends.idxmin().strftime('%Y')
print(f"The year with the highest crime rate is {max_year}.")
print(f"The year with the lowest crime rate is {min_year}.")
plt.show()


# Seasonal Patterns:
# - Group the data by month and analyze the average number of crimes per month over the years

#4.2 Seasonal patterns 
# Seasonal patterns for every month in all the years
df['Month'] = df['DATE OCC'].dt.month
cc_month = df.groupby('Month').size()
plt.figure(figsize=(6, 4))
cc_month.plot(kind='line', marker='o', color='rebeccapurple')
plt.title('Seasonal Patterns in Crime Data')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.show()


# #### Observation
# - The output represents the number of crimes reported for each month across different years from 2020 to 2023. The data is structured to show the month, year, and the corresponding count of crimes for each specific month and year combination. From this data, one can infer the seasonal patterns and trends in crime occurrences over the years. The values illustrate the variations in crime rates throughout different months and years, providing insights into potential patterns or fluctuations that may be occurring annually.
# - In the month of july highest no.of crimes has been recorded over the years.
# - In November less no.of crimes have been recorded over years, this can be due to missing data for the month November and December for the year 2023.
# - Excluding November and December Febraury has least number of crimes.


#Seasonal patterns of crime rate over years 
df['Month'] = df['DATE OCC'].dt.month
df['Year'] = df['DATE OCC'].dt.year
df1 = pd.DataFrame(df.groupby(['Month','Year']).size()).reset_index()
# print(df1)
plt.figure(figsize=(15, 9))
plt.plot(df1[df1['Year']==2020]['Month'],df1[df1['Year']==2020].iloc[:,2], label='2020', marker= 'o')
plt.plot(df1[df1['Year']==2021]['Month'],df1[df1['Year']==2021].iloc[:,2], label='2021', marker= 'o')
plt.plot(df1[df1['Year']==2022]['Month'],df1[df1['Year']==2022].iloc[:,2], label='2022', marker= 'o')
plt.plot(df1[df1['Year']==2023]['Month'],df1[df1['Year']==2023].iloc[:,2], label='2023', marker= 'o')
plt.legend()
plt.title('Seasonal Patterns in Crime Data')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.show()


# #### Observation
# - In all the years the crime rate has been decreased from the month january to febraury.
# - This might be due to less no.of days in the month of febraury than january.
# - The sudden decrease in the year 2023 for the month october is due to partial data for the month.
# - In General there's always a raise in rate of crimes from the month of september to october and decrease in november.

# Most Common Crime Type:
# - Count the occurrences of each crime type and identify the one with the highest frequency.

#4.3 Most common crime type 
top_type = df['Crm Cd Desc'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_type.plot(kind='barh', alpha=0.9)
plt.title('Top 10 Most Common Types of Crime')
plt.xlabel('Number of Occurrences')
plt.ylabel('Crime Type')
plt.show()
print(top_type)


# #### Observation
# - From the data provided, we can infer that "VEHICLE - STOLEN" is the most common crime type, with a significantly higher count of 88,355 reported incidents.

#Trend for the most occured crime
df1 = pd.DataFrame(df[df['Crm Cd Desc'] == 'VEHICLE - STOLEN'].groupby(['Month','Year']).size()).reset_index()
df1['Month_Year'] = df1.apply(lambda row: str(row['Year']) + ' - ' + str(row['Month']), axis=1)
df1.sort_values(by = ['Year','Month'], inplace=True)

plt.figure(figsize=(15, 9))
plt.plot(df1['Month_Year'],df1.iloc[:,2], marker= 'o')
plt.title('Trend in VEHICLE - STOLEN Crime Data')
plt.xlabel('Month - Year')
plt.xticks(rotation = 90)
plt.ylabel('Number of Crimes')
plt.show()

print("New DataFrame")

print(df1)


# #### Observation
# - From the above values, we can observe the trend in the number of "VEHICLE - STOLEN" crimes over the years and months. Some patterns can be inferred from the data, such as:
# - There is a noticeable fluctuation in the number of "VEHICLE - STOLEN" crimes over the months within each year.
# - Generally, there seems to be a higher frequency of "VEHICLE - STOLEN" crimes during the latter months of each year, with some variations between different years.
# - There appears to be a general upward trend in the number of "VEHICLE - STOLEN" crimes from the beginning of 2020 to the end of 2022. However, there seems to be a notable decline in the later part of 2023.
# - The trend reached it's peak point at the same point of time (10-2021).
# - From there it started in a downward trend with no much consistency in the overall trend.

# Regional Differences:
# - Group the data by region or city and compare crime rates between them using descriptive statistics or visualizations.

#4.4 Regional differences 
df2 = pd.DataFrame(df.groupby('AREA NAME').size().reset_index())
df2.columns.values[1] = 'Count'
df2 = df2.sort_values(by ='Count')
plt.figure(figsize=(8, 5))
sns.barplot(data = df2, x ='AREA NAME', y ='Count',palette = 'Blues_d')
plt.title('Crime Rates across different Regions or Cities')
plt.xticks(rotation=90)
plt.show()


df2 = pd.DataFrame(df.groupby('AREA NAME').size().reset_index())
df2.columns.values[1] = 'Count'
df2 = df2.sort_values(by='Count')
print("New DataFrame")
print(df2)


# #### Observations
# - The Central region has the highest count of reported crimes, with 55,567 incidents, followed closely by the 77th Street region with 52,087 incidents.
# - The Foothill and Hollenbeck regions have the lowest counts of reported crimes, with 27,497 and 30,980 incidents, respectively.
# - Foothill has the less no.of crimes.
# - Central has the more no.of crimes.
# - Most of the regions are in the range of 30000 to 40000.

# Correlation with Economic Factors:
# - Collect economic data for the same time frame and use statistical methods like correlation analysis to assess the relationship between economic factors and crime rates.



#4.5 Explore correlations between economic factors and crime rates
economic_data = pd.read_csv("data\economic-indicators.csv") 

print("Econmic data columns")
print(economic_data.columns)


# Load datasets
# crime_data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
# economic_data = pd.read_csv("economic-indicators.csv")

df['YEAR'] = df['DATE OCC'].dt.year

# Merge datasets on 'YEAR' column
merged_data = pd.merge(df, economic_data, on='YEAR', how='inner')

# Calculate total number of crimes for each year
total_crimes = merged_data.groupby('YEAR')['DR_NO'].nunique().tolist()
unemployment_rates = merged_data.groupby('YEAR')['unemp_rate'].mean().tolist()

# Analyze the relationship between total number of crimes and unemployment rate
plt.figure(figsize=(10, 6))
plt.scatter(unemployment_rates, total_crimes)
plt.title('Relationship between Unemployment Rate and Total Number of Crimes')
plt.xlabel('Unemployment Rate')
plt.ylabel('Total Number of Crimes')
for i, year in enumerate(merged_data['YEAR'].unique().tolist()):
    plt.text(unemployment_rates[i], total_crimes[i], str(year), fontsize=10)
plt.show()

# Calculate the Pearson correlation coefficient and the p-value
correlation, p_value = pearsonr(unemployment_rates, total_crimes)
print(f"Pearson Correlation: {correlation:.2f}")
print(f"P-value: {p_value:.2f}")


# A Pearson correlation coefficient of 0.02 and a p-value of 0.98 indicate that there is a very weak positive correlation between the unemployment rate and the total number of crimes. 
# 
# The positive correlation suggests that as the unemployment rate increases, the total number of crimes also slightly increases. However, the correlation is very close to 0, indicating that the relationship is negligible. Therefore, based on the correlation, it appears that the unemployment rate has minimal influence on the total number of crimes in the given dataset.

#4.6 Analyze the relationship between the day of the week and the frequency of certain types of crimes
df['Day of Week'] = df['DATE OCC'].dt.day_name()
plt.figure(figsize=(6, 4))
sns.countplot(x='Day of Week', data=df, palette='Blues', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Frequency of crimes by days of the week')
plt.show()



print(df['Day of Week'].value_counts())


# #### Observation
# - On tuesdays lowest number of crimes have been recorded.
# - On Fridays there's most no.of crimes have been recorded.
# - On the whole the crime rate ranges from 100000 to 120000.
# - From the numerical data, we can observe that the number of crimes reported on Fridays is the highest, with 125,878 occurrences. Saturdays closely follow with 120,615 reported crimes, while Wednesdays, Mondays, Thursdays, Sundays, and Tuesdays have 117,126, 116,894, 116,436, 115,116, and 113,147 reported crimes, respectively. This data suggests a relatively consistent level of criminal activities during weekdays, with a notable increase on Fridays and Saturdays, indicating potential patterns in criminal behavior based on the day of the week.

# # Day of the Week Analysis:
# - Group the data by day of the week and analyze crime frequencies for each day.

#4.6 Analyze the relationship between the day of the week and the frequency of certain types of crimes
top_9_crimes = df['Crm Cd Desc'].value_counts().head(9).index
df['Day of Week'] = df['DATE OCC'].dt.day_name()
z = 1
plt.figure(figsize=(35, 25))
for i in range(9):
    plt.subplot(3,3,z) 
    ax = sns.countplot(x='Day of Week', data=df[df['Crm Cd Desc']==top_9_crimes[i]], palette='Blues', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    # Change the font size of labels
    ax.set_xlabel("Day of Week", fontsize=20)  # Change x-axis label font size
    ax.set_ylabel("Count", fontsize=20)  # Change y-axis label font size
    ax.set_title(top_9_crimes[i], fontsize=20)  # Change title font size

    # Change font size of tick labels on both x and y axes
    ax.tick_params(axis="both", labelsize=15)
    z= z+1
plt.suptitle('Frequency of crimes by days of the week for top 9 crimes', fontsize =25)
plt.show()


# #### Observations
# - Top 9 crimes have been displayed over the weeks.
# - The counts for various types of crimes remain relatively consistent throughout the week, with minor fluctuations between each day.
# - Weekends, specifically Saturdays, tend to have slightly higher reported crime counts for most of the top crimes, including "Vehicle - Stolen," "Battery - Simple Assault," "Theft Plain - Petty ($950 & Under)," "Burglary From Vehicle," "Burglary," "Vandalism - Felony," "Robbery," "Assault with Deadly Weapon, Aggravated Assault," and "Intimate Partner - Simple Assault."
# 
# - Most of the crimes have high frequency on Friday.
# - Theft of Identity has been significantly decreased on the weekends.
# - Some crimes, such as "Battery - Simple Assault" and "Intimate Partner - Simple Assault," show a slightly higher count during the weekend, possibly indicating an association with increased social activities or leisure time.
# - Intimate Partner - simple assault has an increase over the weekends as partners tend to stay at home.
# - Similarly Assault with Deadly weapon has also seen the same trend over the weekends.
# - Vandalism - Felony has recorded it's most no.of crimes on Friday as christians tend to go to church on fridays.
# - Theft Plain - petty has decrased over weekends maybe due to increase no.of shoppers in the store.

# # Impact of Major Events:
# - Identify major events or policy changes during the dataset period and analyze crime rate changes before and after these events.

#4.7 Impact of major events 
import pandas as pd

# Reading the datasets
df_events = pd.read_csv(r"data\Major events file.csv")

df['YEAR'] = df['YEAR'].astype(int)
df_events['YEAR'] = df_events['YEAR'].astype(int)

# Merge the datasets on the 'YEAR' column
merged_data = pd.merge(df, df_events, on='YEAR', how='inner')

# Analyzing the impact of major events on crime
event_counts = merged_data['Major event/policy'].value_counts()
print("Number of occurrences for each major event/policy:")
print(event_counts)

# You can perform further analysis or visualization based on your specific requirements

plt.figure(figsize=(13, 20))
event_counts.plot(kind='barh')  # 'barh' for horizontal bar plot
plt.title('Number of Crimes for Each Major Event/Policy')
plt.xlabel('Number of Crimes')
plt.ylabel('Major Event/Policy')
plt.show()

most_common_event = event_counts.idxmax()
print(f"The event with the most number of crimes is: {most_common_event}")


# # Outliers and Anomalies:
# - Use statistical methods or data visualization techniques to identify dataset outliers 
# and investigate unusual patterns.

#4.8 Outliers and Anomalies
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x='Vict Sex', ax=axes[0, 0])
axes[0, 0].set_title('Gender Distribution')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Count')

sns.boxplot(data=df, x='Vict Age', ax=axes[0, 1])
axes[0, 1].set_title('Boxplot for Victim Age')

sns.histplot(data=df, x='Vict Age', bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram for Victim Age')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

sns.countplot(data=df, x='Vict Descent', ax=axes[1, 1])
axes[1, 1].set_title('Victim Descent Distribution')
axes[1, 1].set_xlabel('Descent')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()



# Dealing with outliers
# Calculate quartiles
Q1 = df['Vict Age'].quantile(0.25)
Q3 = df['Vict Age'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define the upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify and print the outliers
outliers = df[(df['Vict Age'] < lower_bound) | (df['Vict Age'] > upper_bound)]
print(outliers['Vict Age'])
df = df[(df['Vict Age'] != 0) & (df['Vict Age'] < 100)]
df= df[(df['Vict Sex'] != 'H') & (df['Vict Sex'] != '-')]
df= df[(df['Vict Descent'] != '-')]


# # Handiling outliers 
# - The value 120 in the 'Vict Age' column appears to be an outlier based on the calculated quartiles and interquartile range. An age of 120 is unusually high and uncommon in typical demographic distributions. This could be due to data entry errors, missing values, or potentially exceptional cases that need to be investigated further.
# - From the hisogram for age it can be seen that alot of crimes has the age of 0 which is not reasonable and possible hence removing zero 
# - From teh box plot it can be observed that age grater than 100 is outlier
# - From the column description in the origin website, age is only M, F, X and H and - are incosistent data
# - From the column description in the origin website, descent doesnt suppose to have -

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x='Vict Sex', ax=axes[0, 0])
axes[0, 0].set_title('Gender Distribution')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Count')

sns.boxplot(data=df, x='Vict Age', ax=axes[0, 1])
axes[0, 1].set_title('Boxplot for Victim Age')

sns.histplot(data=df, x='Vict Age', bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram for Victim Age')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

sns.countplot(data=df, x='Vict Descent', ax=axes[1, 1])
axes[1, 1].set_title('Victim Descent Distribution')
axes[1, 1].set_xlabel('Descent')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# # Demographic Factors:
# - Analyze the dataset to identify any patterns or correlations between demographic factors (e.g., age, gender) and specific types of crimes.


#4.9 Demographic Factors
plt.figure(figsize=(6, 4))
sns.histplot(data=df[df['Vict Age'] != 0], x='Vict Age', element="step", common_norm=False, color='Blue')
crime_types = df['Crm Cd Desc'].unique()
plt.legend(labels=crime_types, title='Crime Type', title_fontsize='14', loc='upper right')
plt.title('Distribution of Victim Age for Specific Types of Crimes')
plt.xlabel('Victim Age')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Vict Sex', data=df)
plt.title('Relationship between Gender and Specific Types of Crimes')
plt.show()

# correlation_matrix = df[['Vict Age', 'Vict Sex', 'Crm Cd']].corr()
# plt.figure(figsize=(6, 4))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix between Demographic Factors and Crime Types')
# plt.show()

top_9_crimes = df['Crm Cd Desc'].value_counts().head(9).index
df['Day of Week'] = df['DATE OCC'].dt.day_name()
df_age = df[df['Vict Age'] != 0]
z = 1
plt.figure(figsize=(35, 35))
for i in range(9):
    plt.subplot(3,3,z) 
    ax = sns.histplot(x='Vict Age', data=df_age[df_age['Crm Cd Desc']==top_9_crimes[i]],element="bars", bins = 20, common_norm=False)
    # Change the font size of labels
    ax.set_xlabel("Age", fontsize=20)  # Change x-axis label font size
    ax.set_ylabel("Count", fontsize=20)  # Change y-axis label font size
    ax.set_title(top_9_crimes[i], fontsize=20)  # Change title font size
    
    # Change font size of tick labels on both x and y axes
    x_labels = range(0, 100, 5)
    
     # Define the x-labels you want
    plt.xticks(x_labels)
    ax.tick_params(axis="both", labelsize=15)
    z= z+1
plt.suptitle('Distribution of Victim Age for Specific Types of Crimes', fontsize =25)
plt.show()

df['Crm Cd Desc'].value_counts().head(9)


# 
# In the case of "BATTERY - SIMPLE ASSAULT," the data demonstrates a more spread-out distribution, with the peak count appearing at the age of 30.
# 
# "THEFT OF IDENTITY" shows a peak count at the age of 30, indicating that individuals around this age might be more commonly affected by identity theft.
# 
# The distribution for "BURGLARY FROM VEHICLE" is relatively uniform, with a slight peak count at the age of 30.
# 
# "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT" showcases a diverse distribution, with the peak count at the age of between 25 to 30.
# 
# "INTIMATE PARTNER - SIMPLE ASSAULT" indicates a slightly more dispersed age distribution, with the peak count at between 25 to 30
# 
# "VANDALISM - FELONY exhibits a more diverse age distribution, with the peak count at 30 years.
# 
# "THEFT PLAIN - PETTY ($950 & UNDER)" demonstrates a relatively uniform age distribution, with the peak count at the age between 25 to 30.
# 
# In the case of "BURGLARY," the distribution displays a more uniform pattern, with the peak count at the age of 30.
# 
# In the case of "THEFT FROM MOTOR VEHICLE - GRAND," the distribution displays a more uniform pattern, with the peak count at the age of 30.
# 


top_9_crimes = df['Crm Cd Desc'].value_counts().head(9).index
df['Day of Week'] = df['DATE OCC'].dt.day_name()
df_gender = df[(df['Vict Sex'] != 'H') & (df['Vict Sex'] != '-')]
z = 1
plt.figure(figsize=(35, 35))
for i in range(9):
    plt.subplot(3,3,z) 
    ax = sns.countplot(x='Vict Sex', data=df_gender[df_gender['Crm Cd Desc']==top_9_crimes[i]])
    # Change the font size of labels
    ax.set_xlabel("Gender", fontsize=20)  # Change x-axis label font size
    ax.set_ylabel("Count", fontsize=20)  # Change y-axis label font size
    ax.set_title(top_9_crimes[i], fontsize=20)  # Change title font size
    
    # Annotate the values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize =15)
    
    y_labels = range(0, 91000, 10000)
    
     # Define the x-labels you want
    plt.yticks(y_labels)
    
    ax.tick_params(axis="both", labelsize=15)
    z= z+1
plt.suptitle('Distribution of Victim Sex for Specific Types of Crimes', fontsize =25)
plt.show()


# For "VEHICLE - STOLEN," the count of male victims is significantly higher than that of female victims, suggesting that males might be more commonly targeted in these cases.
# 
# "BATTERY - SIMPLE ASSAULT" shows a relatively balanced distribution between male and female victims. The number of female victims is slightly higher compared to male victims.
# 
# "THEFT OF IDENTITY" indicates a higher count of female victims, indicating that females might be more commonly targeted for identity theft compared to males.
# 
# "BURGLARY FROM VEHICLE" exhibits a more balanced distribution between male and female victims, with a slightly higher count of male victims.
# 
# "VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)" demonstrates a substantially higher count of male victims compared to female victims.
# 
# "BURGLARY" shows a higher count of male victims compared to female victims, indicating that males might be more commonly targeted in burglary cases.
# 
# "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT" displays a significantly higher count of male victims compared to female victims, suggesting that males might be more commonly targeted for this type of assault.
# 
# "THEFT PLAIN - PETTY ($950 & UNDER)" showcases a relatively balanced distribution between male and female victims, with a slightly higher count of female victims.
# 
# "INTIMATE PARTNER - SIMPLE ASSAULT" exhibits a substantially higher count of female victims compared to male victims, suggesting that females might be more commonly targeted in cases of intimate partner violence.

# ## Advanced Analysis

# # Predicting Future Trends:
# - Employ time series forecasting methods, such as ARIMA or Prophet, to predict future crime trends based on historical data. Consider incorporating relevant external factors into your models.


df3 = pd.DataFrame(df.groupby(['Month','Year']).size()).reset_index()
df3['Month_Year'] = df3.apply(lambda row: str(row['Year']) + ' - ' + str(row['Month']), axis=1)
df3.columns.values[2] = 'Count'
df3.sort_values(by = ['Year','Month'], inplace=True)
data = df3.loc[:,['Month_Year','Count']]
data.set_index('Month_Year', inplace=True)


data = data[:-1]
# Fit SARIMA model to the data
model = SARIMAX(data['Count'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # You may need to adjust order and seasonal_order
results = model.fit()
future_dates = ['2023-10', '2023-11', '2023-12', '2024-01',
               '2024-02', '2024-03', '2024-04', '2024-05',
               '2024-06', '2024-07', '2024-08', '2024-09']
# Forecast future data for the next year (2023)
forecast_horizon = 12  # Adjust as needed
predict = results.predict(start=data.index[0], end=data.index[-1], dynamic=False)
forecast = results.get_forecast(steps=forecast_horizon)

# Plot the original data and forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Count'], label='Actual Data', marker='o')
plt.plot(data.index[1:], predict[1:], label='Predicted Data', marker='o', linestyle='--', color='orange')
plt.plot(['2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31',
               '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30',
               '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31'], forecast.predicted_mean, marker='o',label='Forecast', linestyle='--', color='orange')

# plt.fill_between(forecast.index, forecast.conf_int()[:, 0], forecast.conf_int()[:, 1], color='orange', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation = 90)
plt.title('Time Series Forecast for 2023')
plt.legend()

plt.grid(True)

plt.show()


# - The message "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH" means that the reduction in the objective function value is less than the specified threshold (likely the default machine precision).
# 
# - The total number of iterations performed is 22, which includes function evaluations and other steps involved in the optimization process.
# 
# - The algorithm made 27 function evaluations during the optimization process, suggesting that it probed the objective function at different points in the parameter space to find the optimal solution.
# 
# - The projected gradient at the final iteration is very small (5.130D-05), indicating that the gradient is close to  zero, and the algorithm is close to the optimal solution.
# 
# - The final function value achieved by the SARIMAX model is 5.7794127462513689, which likely corresponds to the minimized error or loss function.
# 
# - Overall, the output suggests that the SARIMAX model has been successfully fitted to the data, and the optimization process has converged, indicating that the model parameters have been estimated effectively.