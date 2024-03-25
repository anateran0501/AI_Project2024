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


movies = pd.read_csv(r"C:\Users\shuba\Downloads\moviesmetadata.csv")
ratings = pd.read_csv(r"C:\Users\shuba\Downloads\ratings.csv\ratings.csv")



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


# Drop the 'belongs_to_collection' column
cleaned_df = cleaned_df.drop('belongs_to_collection', axis=1)

# Verify that the column has been dropped
print(cleaned_df.columns)


# In[12]:


cleaned_df.count()


# In[13]:


import pandas as pd

# Directory path where you want to save the file
directory_path = r"C:\Users\shuba\Desktop\AI PROJECT"

# File name
output_file = "Cleaned_Dataset.csv"

# Concatenate directory path with file name
output_path = directory_path + "\\" + output_file

# Save the DataFrame to the specified path
cleaned_df.to_csv(output_path, index=False)

print("Cleaned DataFrame saved to CSV file successfully.")


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged dataset
cleaned_df = pd.read_csv(r"C:\Users\shuba\Desktop\AI PROJECT\Cleaned_Dataset.csv")

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


# In[15]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset with engineered features
feature_engineered_df = pd.read_csv(r"C:\Users\shuba\Downloads\feature_engineered_dataset.csv")

# Extract temporal features
feature_engineered_df['timestamp'] = pd.to_datetime(feature_engineered_df['timestamp'])
feature_engineered_df['day_of_week'] = feature_engineered_df['timestamp'].dt.dayofweek
feature_engineered_df['month'] = feature_engineered_df['timestamp'].dt.month
feature_engineered_df['hour'] = feature_engineered_df['timestamp'].dt.hour

# User activity features
user_activity = feature_engineered_df.groupby('userId').agg({
    'rating': ['count', 'mean', 'std']  # Number of ratings, average rating, and standard deviation of ratings
}).reset_index()
user_activity.columns = ['userId', 'num_ratings', 'avg_rating', 'std_rating']
feature_engineered_df = pd.merge(feature_engineered_df, user_activity, on='userId', how='left')

# Movie popularity features
movie_popularity = feature_engineered_df.groupby('movieId').agg({
    'rating': ['count', 'mean']  # Number of ratings and average rating for each movie
}).reset_index()
movie_popularity.columns = ['movieId', 'num_ratings_movie', 'avg_rating_movie']
feature_engineered_df = pd.merge(feature_engineered_df, movie_popularity, on='movieId', how='left')

# Genre encoding (example: one-hot encoding for top N genres)
top_genres = feature_engineered_df['genres'].str.split(',', expand=True).stack().value_counts().head(10).index.tolist()
for genre in top_genres:
    feature_engineered_df[f'genre_{genre}'] = feature_engineered_df['genres'].apply(lambda x: 1 if genre in x else 0)

# Save the dataset with engineered features
feature_engineered_df.to_csv(r"C:\Users\shuba\Downloads\feature_engineered_dataset.csv", index=False)


# In[16]:


get_ipython().run_line_magic('pip', 'install scikit-surprise')


# In[17]:


from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import time

# Load the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(cleaned_df[['userId', 'movieId', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Select and train the model
start_fit_time = time.time()
model = SVD()
model.fit(trainset)
end_fit_time = time.time()

fit_time = end_fit_time - start_fit_time
print("Fit time:", fit_time)

# Make predictions on the test set
start_test_time = time.time()
predictions = model.test(testset)
end_test_time = time.time()

test_time = end_test_time - start_test_time
print("Test time:", test_time)

# Evaluate the model
accuracy.rmse(predictions)
accuracy.mae(predictions)

# Example: Get top-N recommendations for a user
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all movie IDs
    all_movie_ids = cleaned_df['movieId'].unique()
    
    # Exclude movies already rated by the user
    rated_movie_ids = cleaned_df.loc[cleaned_df['userId'] == user_id, 'movieId'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for unrated movies
    unrated_predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort the predictions by estimated rating in descending order
    top_n_predictions = sorted(unrated_predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get movie titles corresponding to the top-N predicted movie IDs
    top_n_movie_titles = [cleaned_df.loc[cleaned_df['movieId'] == pred.iid, 'title'].iloc[0] for pred in top_n_predictions]
    
    return top_n_movie_titles

# Example: Get top 10 recommendations for user 1
user_id = 1
top_n_recommendations = get_top_n_recommendations(user_id)
print(f"Top 10 recommendations for user {user_id}:")
print(top_n_recommendations)


# In[38]:


import matplotlib.pyplot as plt

# Data
fit_time = fit_time
test_time = test_time

# Plotting fit time and test time with narrower bars
labels = ['Fit Time', 'Test Time']
values = [fit_time, test_time]

plt.figure(figsize=(7, 6))
bars = plt.bar(labels, values, color=['blue', 'green'], width=0.2)  # Adjust the width of bars here
plt.xlabel('Metrics')
plt.ylabel('Values in sec')
plt.title('Evaluation Metrics')

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

plt.show()


# In[46]:


import matplotlib.pyplot as plt

# Data
rmse = rmse
mae = mae

# Plotting RMSE and MAE with narrower bars
labels = ['RMSE', 'MAE']
values = [rmse, mae]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['red', 'orange'], width=0.2)
plt.xlabel('Metrics')
plt.ylabel('Ratings from 0.5-5')
plt.title('Evaluation Metrics')

# Set y-axis limits to 0.5 and 5
plt.ylim(0.5, 5)

# Set y-axis ticks from 0.5 to 5 with increments of 0.5
plt.yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

plt.show()


# In[ ]:




