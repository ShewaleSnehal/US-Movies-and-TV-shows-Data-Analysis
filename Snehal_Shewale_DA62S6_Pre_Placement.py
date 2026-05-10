#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # Array Processing 
import pandas as pd # Data Ananlysis 
from sklearn.impute import KNNImputer # Data Imputation
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sb # Data Visualization
import scipy.stats as st # Statistical Analysis


# In[2]:


Movies_Data=pd.read_csv('MoviesOnStreamingPlatforms_updated.csv') # read csv file
Movies_Data


# In[3]:


Movies_Data.shape


# ###  Country With Highest Counts of Movies

# In[4]:


top_country = Movies_Data['Country'].mode()[0]
print("Country with the highest number of movies:", top_country)

# United States has the highest number of movies listed in the dataset 


# ### Filter Top Most Country Data (United States )

# In[5]:


US_Movies_Data=Movies_Data[Movies_Data['Country']=='United States']
US_Movies_Data


# In[6]:


US_Movies_Data.shape  


# ### Data Cleaning

# In[7]:


US_Movies_Data.info() # 


# In[8]:



US_Movies_Data['Rotten Tomatoes'] = US_Movies_Data['Rotten Tomatoes'].str.replace('%', '').astype(float)


# In[9]:


US_Movies_Data['Rotten Tomatoes'].unique()


# In[10]:


US_Movies_Data.info()


# In[11]:


US_Movies_Data.isnull().sum()


# In[12]:


knn_imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ["IMDb", "Rotten Tomatoes", "Runtime"]
US_Movies_Data[columns_to_impute] = knn_imputer.fit_transform(US_Movies_Data[columns_to_impute])


# In[13]:


US_Movies_Data.isnull().sum()


# In[14]:


US_Movies_Data['Age'].unique()


# In[15]:


US_Movies_Data['Age'].mode()


# In[16]:


US_Movies_Data['Age'].fillna('18+',inplace=True)


# In[17]:


US_Movies_Data.dropna(inplace=True)


# In[18]:


US_Movies_Data.isnull().sum()


# In[19]:


US_Movies_Data.info()


# In[20]:


# Total 8416 rows and 15 columns aviable in US_Movies_Data


# ### Highest IMDb Rating TOP 10 Movies 

# In[21]:


sorted_data = US_Movies_Data.sort_values(by='IMDb', ascending=False)
top_rated_movies = sorted_data.head(10)  
print(top_rated_movies[['Title', 'IMDb']])


# 'Steven Banks: Home Entertainment Center' and 'Square one' are the movie with the highest IMDb rating, which is 9.3.
# The movie with the lowest IMDb rating is "Operation Toussaint: Operation Underground Railroad and the Fight to End Modern Day Slavery," which has a rating of 8.8.


# ###  What platform has the highest movie count available?

# In[22]:


# Which platform offers the highest number of movies and TV shows in the dataset?
platform_counts = US_Movies_Data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].sum()
most_content_platform = platform_counts.idxmax()
print('TOP Movie Content Platform: ',most_content_platform)

# Prime Video has the largest movie library among available streaming platforms.


# ### Number of Movies and TV Shows by Platform

# In[23]:


platform_counts = US_Movies_Data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].sum()
platform_counts = platform_counts.sort_values(ascending=False)
print(platform_counts)
plt.figure(figsize=(10, 6))
platform_counts.plot(kind='bar')
plt.title('Number of Movies and TV Shows by Platform')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.show()

# Prime Video has the highest number of movies among available platforms, and Disney+ has the least.


# In[24]:


# What is the distribution of IMDb ratings for content available on Netflix and Prime Video?
imdb_ratings_by_platform = US_Movies_Data.groupby(['Prime Video','Netflix'])[['IMDb']].describe().T
print(round(imdb_ratings_by_platform,2))
US_Movies_Data.boxplot(column='IMDb', by=(['Prime Video','Netflix']), vert=True)
plt.suptitle('')
plt.show()

# Netflix averages a 6.26 IMDb rating for movies and TV shows, while Prime Video shows average 5.58, suggesting US customers prefer Netflix.
# The highest IMDb rating for Prime Video is 9.30, while Netflix's maximum rating is 8.70.


# In[25]:


# How does the distribution of Rotten Tomatoes ratings vary across the platforms?
rotten_tomatoes_by_platform = US_Movies_Data.groupby(['Prime Video','Netflix'])[['Rotten Tomatoes']].describe().T
print(round(rotten_tomatoes_by_platform),2)
US_Movies_Data.boxplot(column='Rotten Tomatoes', by=(['Prime Video','Netflix']), vert=True)
plt.suptitle('')
plt.show()

# Netflix movies typically have Rotten Tomatoes ratings ranging from 46% to 85%.
# Prime Video movies usually have Rotten Tomatoes ratings falling between 40% and 77%, suggesting that Netflix movies receive more positive reviews.


# In[26]:


# What is the age distribution for movies and TV shows on Prime Video & Netflix platforms?


# In[27]:


age_distribution_by_platform = US_Movies_Data.groupby(['Prime Video','Netflix'])['Age'].value_counts().unstack(fill_value=0)
print(age_distribution_by_platform)


age_distribution_by_platform.plot(kind='bar', label='platform')
plt.title('Age Distribution by Platform')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Most adults aged 18+ prefer watching movies and TV shows on Prime Video.
# Similarly, Netflix is the choice for many adults aged 18+.


# In[28]:


# How does the number of movies and TV shows change over the years?


# In[29]:


content_count_by_year = US_Movies_Data['Year'].value_counts().sort_index(ascending=False)
print(content_count_by_year)
plt.figure(figsize=(10, 6))
plt.title('Number of Movies and TV Shows Over the Years')
plt.xlabel('Year')
plt.ylabel('Count')
US_Movies_Data.groupby('Year').size().plot(kind='line')
plt.show()

# Most adults aged 18+ prefer watching movies and TV shows on Prime Video.
# Similarly, Netflix is the choice for many adults aged 18+.


# In[30]:


# What is the average IMDb rating by age category?


# In[31]:


average_imdb_by_age =US_Movies_Data.groupby('Age')['IMDb'].mean()
print(average_imdb_by_age)
plt.figure(figsize=(10, 6))
plt.title('Average IMDb Rating by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Average IMDb Rating')
average_imdb_by_age.plot(kind='bar')
plt.show()

# The average rating for all categories falls within the range of 5.2 to 6.08.


# In[32]:


# How many movies on each platform have an IMDb rating above 8.0?


# In[33]:


high_rated_counts = US_Movies_Data[US_Movies_Data['IMDb'] > 8.0][['Prime Video','Netflix', 'Disney+', 'Hulu']].sum()
print("Number of highly rated movies (IMDb > 8.0) on each platform:")
print(high_rated_counts)
plt.figure(figsize=(10, 6))
plt.title('Count of IMDb rating for each Platform')
plt.xlabel('Platform')
plt.ylabel('IMDb Rating count')
high_rated_counts.plot(kind='bar')
plt.show()


# Prime Video has the most movies with IMDb ratings above 8, totaling 158.
# Following that, Netflix has 37, Disney+ has 17, and Hulu has the fewest with 12.


# In[34]:


# What is the distribution of runtime for movies and TV shows?


# In[35]:


runtime_distribution = US_Movies_Data['Runtime'].describe()
print(runtime_distribution)
plt.title('Distribution of Runtime')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Count')
US_Movies_Data['Runtime'].plot(kind='hist', bins=20)
plt.show()

# The runtime of movies typically falls between 79 to 98 minutes, with a maximum of 328 minutes and a minimum of 2 minutes.
# The average runtime for these movies is approximately 88 minutes.


# In[36]:


# How does IMDb rating vary with runtime?


# In[37]:


imdb_by_runtime = US_Movies_Data.groupby(pd.cut(US_Movies_Data['Runtime'], bins=5))['IMDb'].mean().round()
imdb_by_runtime


# In[38]:


plt.figure(figsize=(10, 6))
plt.title('IMDb Rating vs. Runtime')
plt.xlabel('Runtime (minutes)')
plt.ylabel('IMDb Rating')
plt.scatter(US_Movies_Data['Runtime'], US_Movies_Data['IMDb'], alpha=0.3)
plt.show()

# The maximum spread of the data cluster occurs within the IMDb ratings range of 4 to 8 and the runtime range of 50 to 120, 
# although some outliers are present in the data.


# In[39]:


# What is the distribution of genres?


# In[40]:


genre_counts = US_Movies_Data['Genres'].str.split(',').explode().str.strip().value_counts().head()
print(genre_counts)
plt.figure(figsize=(12, 6))
plt.title('Distribution of Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
genre_counts.plot(kind='bar')
plt.show()

# Drama genre has the highest movie count, while horror has the fewest.
# Most people from the US are interested in the Drama and Comedy genres.


# In[41]:


# How does IMDb rating vary across Top 5 genres?


# In[42]:


imdb_by_genre = US_Movies_Data.groupby('Genres')['IMDb'].mean().sort_values(ascending=False).head()
imdb_by_genre


# The genres Short, Drama, and Thriller have the highest IMDb rating, averaging 8.8.
# On the other hand, the genres Documentary, Biography, Music, and News have the lowest IMDb rating, averaging 8.5.


# In[43]:


# How do IMDb ratings vary for movies Top 10 languages?


# In[44]:


imdb_by_language = US_Movies_Data.groupby('Language')['IMDb'].mean().sort_values(ascending=False).head(10)
imdb_by_language

# English,Japanese,Xhosa,German  this langauage contain highest IMDB Rating means most of the people like to watch movies and TV shows with this languages.


# In[45]:


# How does the IMDb rating change with age category?


# In[46]:


imdb_by_age = US_Movies_Data.groupby('Age')['IMDb'].mean()
imdb_by_age

# IMDb ratings for all age groups fall within the range of 5.2 to 6.08.


# In[47]:


# How many Movies comes under Positive reviews?


# In[48]:


Positive_Reviews=US_Movies_Data[US_Movies_Data['Rotten Tomatoes']>=60]
print('Positive_Reviews_movie_Count:',Positive_Reviews['Rotten Tomatoes'].count())

# Approximately 50% of people provide positive reviews for both movies and TV shows.


# In[49]:


# What is the year-wise distribution of movies on each platform?
yearly_distribution = US_Movies_Data.groupby(['Year', 'Netflix', 'Hulu', 'Disney+', 'Prime Video'])['Title'].count().unstack(fill_value=0)
yearly_distribution.plot(kind='line', figsize=(12, 6))
plt.title("Yearly Distribution of Movies on Each Platform")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.legend(title="Platform")
plt.show()

# The number of movies available on all platforms shows a gradual increase by year.


# In[50]:


# What is the most common genre among the listed movies?


# In[51]:


common_genre = US_Movies_Data['Genres'].mode()[0]
print("Most common genre:", common_genre)

# The most common genre among the listed movies is "Documentary."


# In[52]:


# Find Spread of data for all Platforms?


# In[53]:


plt.figure(figsize=(15,5))  

plt.subplot(2, 2, 1)
plt.title('Prime Video')
sb.kdeplot(US_Movies_Data['Prime Video'], color='yellow', shade=True)

plt.subplot(2, 2, 2)
plt.title('Netflix')
sb.kdeplot(US_Movies_Data['Netflix'], color='green', shade=True)

plt.subplot(2, 2, 3)
plt.title('Hulu')
sb.kdeplot(US_Movies_Data['Hulu'], color='magenta', shade=True)

plt.subplot(2, 2, 4)
plt.title('Disney+')
sb.kdeplot(US_Movies_Data['Disney+'], color='red', shade=True)
plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()


# In Prime Video, most of the data is distributed between 0.8 to 1.2, with less in the range of -0.2 to 0.2.
# In Netflix, Hulu, and Disneyx, the majority of the data is distributed between -0.2 to 0.2, with less in the range of 0.8 to 1.2.


# In[54]:


US_Movies_Data.to_csv("US_Country_Movies_Data.csv", index=False)   # Save file in CSV format

