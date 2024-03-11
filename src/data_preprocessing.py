# ## 1. Data Acquisition

#importing all the necessary libraries
import pandas as pd

df = pd.read_csv('data/Crime_Data_from_2020_to_Present.csv') #Reading the dataset
print("Displaying the dataset:\n",df) #displaying the dataset


# ## 2. Data Inspection



print("Displaying the dataset with all columns and upto 5 rows:\n", df.head(5)) #displaying the dataset with all columns and upto 5 rows


print("Displaying the bottom 5rows with all the columns:\n",df.tail(5)) #displaying the bottom 5rows with all the columns

print("Displaying all the information about the dataset")
print(df.info()) #displaying all the information about the dataset

print("Displaying the summary of each columns")
print(df.describe(include="all")) #displaying the summary of each columns

print("Displaying the datatypes of dataset")
print(df.dtypes) #displaying the datatypes of dataset

print("Displaying the columns names")
print(df.columns) #displaying the columns names


# #### Column Name - Description
# - DR_NO - Division of Records Number: Official file number made up of a 2 digit year, area ID, and 5 digits
# - Date Rptd - MM/DD/YYYY
# - DATE OCC - MM/DD/YYYY
# - TIME OCC - In 24 hour military time.
# - AREA - The LAPD has 21 Community Police Stations referred to as Geographic Areas within the department. These Geographic Areas are sequentially numbered from 1-21.
# - AREA NAME - The 21 Geographic Areas or Patrol Divisions are also given a name designation that references a landmark or the surrounding community that it is responsible for. For example 77th Street Division is located at the intersection of South Broadway and 77th Street, serving neighborhoods in South Los Angeles.
# - Rpt Dist No - A four-digit code that represents a sub-area within a Geographic Area. All crime records reference the "RD" that it occurred in for statistical comparisons. Find LAPD Reporting Districts on the LA City GeoHub at http://geohub.lacity.org/datasets/c4f83909b81d4
# - Part 1-2
# - Crm Cd - Indicates the crime committed. (Same as Crime Code 1)
# - Crm Cd Desc - Defines the Crime Code provided.
# - Mocodes - Modus Operandi: Activities associated with the suspect in commission of the crime.See attached PDF for list of MO Codes in numerical order. https://data.lacity.org/api/views/y8tr-7khq/files/3a967fbd-f210-4857-bc52-60230efe256c?download=true&filename=MO%20CODES%20(numerical%20order
# - Vict Age - Two character numeric
# - Vict Sex - F - Female M - Male X - Unknown
# - Vict Descent -	Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian
# - Premis Cd	- The type of structure, vehicle, or location where the crime took place.
# - Premis Desc - Defines the Premise Code provided.
# - Weapon Used Cd - The type of weapon used in the crime.
# - Weapon Desc - Defines the Weapon Used Code provided.
# - Status - Status of the case. (IC is the default)
# - Status Desc - Defines the Status Code provided.
# - Crm Cd 1 - Indicates the crime committed. Crime Code 1 is the primary and most serious one. Crime Code 2, 3, and 4 are respectively less serious offenses. Lower crime class numbers are more serious.
# - Crm Cd 2 - May contain a code for an additional crime, less serious than Crime Code 1.
# - Crm Cd 3 - May contain a code for an additional crime, less serious than Crime Code 1.
# - Crm Cd 4 - May contain a code for an additional crime, less serious than Crime Code 1.
# - LOCATION - Street address of crime incident rounded to the nearest hundred block to maintain anonymity.
# - Cross Street - Cross Street of rounded Address
# - LAT - Latitude
# - LON - Longtitude

# # 3. Data Cleaning

print("Counting the no.of null values present in each column")

print(df.isnull().sum()) #counting the no.of null values present in each column

print("Checking the percent of data that is missing")
print((df.isnull().sum()/df.shape[0])*100) #checking the percent of data that is missing


# #### Observation
# - For handling categorical columns like "Mocodes", "Vict Sex", and "Vict Descent" we use the high frequency values in the column to fill in the null values.
# 
# - Similarly for handling numerical columns like "Premis Cd" and "Crm Cd 1" we use the high frequency values in the column to fill in the null values.
# 
# - For the rest of the columns there's more than 50% missing values, which are columns "Weapon Used Cd", "Weapon Desc", "Crm Cd 2   ", "Crm Cd 3", "Crm Cd 4" and "Cross Street",  we can drop the columns since they are redundant data and replacing the missing values in those columns with any data handling would cause some miscellaneous results, which is why we are dropping them off.


# Handle missing data
# Handling categorical columns
df['Mocodes'].fillna(df['Mocodes'].mode()[0], inplace=True)
df['Vict Sex'].fillna(df['Vict Sex'].mode()[0], inplace=True)
df['Vict Descent'].fillna(df['Vict Descent'].mode()[0], inplace=True)

# Handling numerical columns
df['Premis Cd'].fillna(df['Premis Cd'].mode()[0], inplace=True)
df['Crm Cd 1'].fillna(df['Crm Cd 1'].mode()[0], inplace=True)

# Handling text columns
df['Premis Desc'].fillna(df['Premis Desc'].mode()[0], inplace=True)

# Drop specified columns
columns_to_drop = ['Weapon Used Cd', 'Weapon Desc', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
df.drop(columns_to_drop, axis=1, inplace=True)

# Verify if columns have been dropped
print("Columns after dropping specified columns: \n", df.columns)

missing_data_after = df.isnull().sum() #checking if there is any missing data
print("Missing Data After Handling:\n", missing_data_after)

print("Checking if there is any duplicate data present in the dataset")
print(df.duplicated().sum()) #Checking if there is any duplicate data present in the dataset


# There aren't any duplicate rows present in this dataset.

#Converting the object datatype of Date Rptd, DATE OCC into date time object
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Custom function to pad single-digit hours and minutes
#Converting the ineteger datatype into time object
def pad_time(time_str):
    time_str = str(time_str).zfill(4)  # Pad with leading zeros
    hours = int(time_str[:2])
    minutes = int(time_str[2:])
    if hours > 23 or minutes > 59:
        return pd.NaT  # Invalid time, return Not-a-Time
    return pd.to_datetime(f"{hours:02}:{minutes:02}", format='%H:%M').time()

# Apply the custom function and convert to datetime
df['TIME OCC'] = df['TIME OCC'].apply(pad_time)

df['Date Rptd'] #displaying converted datatype (date time object)


df['DATE OCC'] #displaying converted datatype (date time object)


# time conversion
#displaying converted datatype (time object)
df['TIME OCC'].head(20) 

# #As these are codes these are supposed to be integers not floats
df['Premis Cd'] = df['Premis Cd'].astype(int) 
df['Crm Cd 1'] = df['Crm Cd 1'].astype(int)


print("Data types after conversion: \n", df.dtypes) # Converted datatypes

df_encoded= pd.get_dummies(df, columns=['Vict Descent','Status','Vict Sex']) #Encoding the categorical variables

print("Displaying the new dataset with encoded categorical variables")
print(df_encoded.head()) #displaying the new dataset with encoded categorical variables

df_encoded.to_csv("data\data_preprocessing.csv")

print("Data Preprocessing completed")
# #### Encoding
# - One hot encoding is used on the columns 'Vict Descent','Status','Vict Sex'.
# - For which the unique values in these columns turns into individual columns and the column value becomes one when it's corresponding rows in the column have the same value as the column. 
# - One hot encoding is used because there are more than two values and if each value is given a individual number model will assume there is heirarchy in the column and gives more weightage to the category with higher value with one hot encoding this can be prevented.
# - AREA NAME has not been changed because the column area represents numerical version of Area name.
# - Crm Cd Desc has not been changed because the column Crm Cd represents numerical version of Crm Cd Desc.
# - Premis Desc has not been changed because the column Premis Cd represents numerical version of Premis Desc.
# - Status Desc has not been changed because column Status represents either numerical version or Status Desc.