Excellent\! Using a real-world dataset from Kaggle will make the experiment much more practical and interesting. We'll modify Experiment 2 to use a popular Kaggle dataset, the **IMDB Movie Reviews** dataset, which is perfect for text preprocessing and sentiment analysis. This will also make the subsequent Fine-tuning experiment (Experiment 8) more realistic.

The dataset can be easily accessed directly with the `datasets` library from Hugging Face, which is often used with Kaggle data.

### Preprocess and Clean the IMDB Movie Reviews Dataset for a Generative AI Application

#### What and Why for each cell:

#### Cell 1: Setup and Data Loading

**What:**

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the IMDB dataset from the Hugging Face Hub
# The 'datasets' library makes this a one-liner for many popular datasets
dataset = load_dataset("imdb")

# Convert the training split to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

print("Original Dataset (first 5 rows):")
print(df.head())
print("\nDataset Information:")
df.info()
```

**Why:**
This cell sets up our environment and loads the actual data from Kaggle's public datasets.

  - We import the necessary libraries: `pandas` for data manipulation, `numpy` for numerical operations, `datasets` to easily load the IMDB dataset, and `re` for regular expressions for text cleaning. We also import `nltk` for its pre-built text processing tools like stopwords.
  - The `load_dataset("imdb")` function automatically downloads and loads the IMDB dataset, which is a standard choice for sentiment analysis. It's a great example of how to access a widely-used Kaggle dataset without manual download/upload steps.
  - We convert the `train` split of the dataset into a pandas DataFrame. This is often the preferred format for exploratory data analysis and for applying cleaning functions, as pandas has powerful and intuitive tools for these tasks.
  - `df.head()` and `df.info()` give us a quick look at the data's structure, columns, and data types, which helps us plan our preprocessing steps.

#### Cell 2: Handling Missing Data and Duplicates

**What:**

```python
# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()
print("\nDataset shape after dropping missing values:", df.shape)

# Check for duplicate entries
print("\nNumber of duplicate rows before cleaning:", df.duplicated().sum())

# Drop duplicate rows
df = df.drop_duplicates()
print("Number of duplicate rows after cleaning:", df.duplicated().sum())
print("Dataset shape after dropping duplicates:", df.shape)
```

**Why:**
This is a critical first step for any real-world dataset.

  - We check for and handle missing data, a common issue in raw datasets. `dropna()` is a simple and effective way to remove any rows that contain missing values.
  - We also check for and remove duplicate reviews. In a large text dataset like IMDB, duplicates can arise from data collection errors or repetition, and they can skew a model's training. `drop_duplicates()` ensures each review is unique.
  - By printing the dataset's shape before and after these operations, we can see exactly how many data points were removed.

#### Cell 3: Text Cleaning (Preprocessing)

**What:**

```python
# Define a comprehensive text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags (common in web-scraped data like this)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords (common words like 'a', 'the', 'is')
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)

    return text

# Apply the cleaning function to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)

print("\nOriginal text vs. Cleaned text (first 5 rows):")
print(df[['text', 'clean_text']].head())
```

**Why:**
Text data needs to be normalized before a model can learn from it effectively. This is the most detailed part of the preprocessing for a text-based Generative AI model.

  - We define a single, comprehensive function `clean_text()` to perform several cleaning tasks in one go.
  - **Lowercasing:** Standardizes the text.
  - **HTML Tag Removal:** The IMDB reviews were likely scraped from a website, so they often contain HTML tags like `<br />`. We use a regular expression (`re.sub`) to remove them.
  - **Punctuation and Number Removal:** We keep only alphabetic characters and spaces. This simplifies the vocabulary and removes noise.
  - **Whitespace Normalization:** Cleans up multiple spaces and leading/trailing spaces.
  - **Stopword Removal:** `stopwords` are very common words that often don't carry much meaning for the sentiment or topic of a sentence. Removing them reduces the vocabulary size and helps the model focus on more important words.
  - The output shows a side-by-side comparison of the original and cleaned text, making the impact of the cleaning steps clear.

#### Cell 4: Tokenization and Preparing Data for a Model

**What:**

```python
# Tokenization (splitting sentences into words)
df['tokens'] = df['clean_text'].apply(lambda x: x.split())

# We can now map the 'label' column (0 or 1) to meaningful strings
label_map = {0: "negative", 1: "positive"}
df['sentiment'] = df['label'].map(label_map)

print("\nDataset after tokenization and label mapping (first 5 rows):")
print(df[['clean_text', 'tokens', 'label', 'sentiment']].head())

# Save the preprocessed data to a new CSV file
df.to_csv("preprocessed_imdb_reviews.csv", index=False)
print("\nPreprocessed data saved to 'preprocessed_imdb_reviews.csv'")
```

**Why:**
This cell finalizes the data preparation and makes it ready for a model.

  - We perform **tokenization**, which is the process of splitting the cleaned text into a list of words. This is a fundamental step for all NLP models.
  - We create a human-readable `sentiment` column from the numerical `label` (0 or 1). This is useful for visualization and interpreting model outputs later.
  - Finally, we save the fully preprocessed DataFrame to a new CSV file. This is a good practice as it saves time by not having to re-run the cleaning steps every time we want to use the data for a new experiment (e.g., in a model training notebook like Experiment 8). The data is now in a clean, structured, and ready-to-use format.
