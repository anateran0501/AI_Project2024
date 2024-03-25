#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)


# In[2]:


movies = pd.read_csv(r"C:\Users\anate\Desktop\Code\moviesmetadata.csv")
ratings = pd.read_csv(r"C:\Users\anate\Desktop\Code\ratings.csv")



# In[3]:


# Check column names
print(movies.columns)

# Display DataFrame
print(movies)



# In[4]:


# Handle missing values
# For movies dataset
movies.dropna(inplace=True)  # Drop rows with any missing values
# For ratings dataset
ratings.dropna(inplace=True)  # Drop rows with any missing values


# In[5]:


# Identify non-numeric values in the 'id' column
non_numeric_values = movies['id'][~movies['id'].str.isdigit()]
print(non_numeric_values)

# Handle non-numeric values (e.g., replace them with NaN)
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

# Drop rows with NaN values in the 'id' column
movies.dropna(subset=['id'], inplace=True)

# Convert the 'id' column to integers
movies['id'] = movies['id'].astype('int64')

# Now you can proceed with the merge
df = movies.merge(ratings, how="left", left_on="id", right_on="movieId")


# In[6]:


movies.rename(columns={'id': 'movieId'}, inplace=True)

# Merge the DataFrames
df = movies.merge(ratings, how="left", on="movieId")
df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.count()


# In[10]:


cleaned_df = df.dropna()


# In[11]:


cleaned_df.count()


# In[13]:


# Directory path where you want to save the file
directory_path = r"C:\Users\91628\Desktop\AI Project\files"

# File name
output_file = "cleaned_dataset.csv"

# Concatenate directory path with file name
output_path = directory_path + "\\" + output_file

# Save the DataFrame to the specified path
cleaned_df.to_csv(output_path, index=False)

print("Cleaned DataFrame saved to CSV file successfully.")


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged dataset
cleaned_df = pd.read_csv('cleaned_dataset.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(cleaned_df.info())

# Summary statistics for numerical variables
print("\nSummary Statistics for Numerical Variables:")
print(cleaned_df.describe())

# Correct data types if necessary (convert 'release_date' to datetime)
cleaned_df['release_date'] = pd.to_datetime(cleaned_df['release_date'])

# Summary statistics for categorical variables
print("\nSummary Statistics for Categorical Variables:")
print(cleaned_df.describe(include=['object']))

# Univariate Analysis
# Histogram of numerical variables
cleaned_df.hist(figsize=(15, 12))
plt.show()

# Step 4: Bivariate Analysis
# Scatter plot of numerical variables
sns.scatterplot(x='budget', y='revenue', data=cleaned_df)
plt.show()

# Step 5: Multivariate Analysis
# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Step 6: Additional Analysis
# Time series plot of rating over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='release_date', y='rating', data=cleaned_df)
plt.show()


# In[ ]:




