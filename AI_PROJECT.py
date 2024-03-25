#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import pandas as pd

# Configure display options for pandas DataFrame
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

# Read movies and ratings datasets
movies = pd.read_csv(r"C:\Users\anate\Desktop\Code\moviesmetadata.csv")
ratings = pd.read_csv(r"C:\Users\anate\Desktop\Code\ratings.csv")

# Check column names of movies dataset
print(movies.columns)

# Display movies dataset
print(movies)

# Handle missing values in movies and ratings datasets
movies.dropna(inplace=True)  # Drop rows with any missing values in movies dataset
ratings.dropna(inplace=True)  # Drop rows with any missing values in ratings dataset

# Identify non-numeric values in the 'id' column of movies dataset
non_numeric_values = movies['id'][~movies['id'].str.isdigit()]
print(non_numeric_values)

# Handle non-numeric values in the 'id' column (e.g., replace them with NaN)
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

# Drop rows with NaN values in the 'id' column of movies dataset
movies.dropna(subset=['id'], inplace=True)

# Convert the 'id' column to integers
movies['id'] = movies['id'].astype('int64')

# Merge movies and ratings datasets based on the 'id' and 'movieId' columns
df = movies.merge(ratings, how="left", left_on="id", right_on="movieId")

# Rename the 'id' column to 'movieId' in movies dataset
movies.rename(columns={'id': 'movieId'}, inplace=True)

# Merge movies and ratings datasets based on the 'movieId' column
df = movies.merge(ratings, how="left", on="movieId")

# Display the resulting DataFrame
df.head()
