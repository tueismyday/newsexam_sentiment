import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import seaborn as sns
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import os
from itertools import chain

#######################################
# Loading the dataset
#######################################

file_path = "data/reddit_wsb.csv"
try:
    print("\nLoading data.")
    reddit_df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Please check the file path and try again.")
    exit(1)

#######################################
# Basic Cleaning
#######################################

# Drop rows with missing URL
reddit_df.dropna(subset=['url'], inplace=True)

#######################################
# Extract stock mentions and remove posts without stock mentions
#######################################

def extract_stocks(text):
    if not isinstance(text, str):  # Handle non-string inputs
        return []
    return [stock[1:] for stock in re.findall(r'\$[A-Z]{1,5}\b', text)]

print("\nExtracting stocks from post titles and bodies.")
reddit_df['stocks_from_title'] = reddit_df['title'].apply(extract_stocks)
reddit_df['stocks_from_body'] = reddit_df['body'].apply(extract_stocks)

reddit_df['mentioned_stocks'] = reddit_df['stocks_from_title'] + reddit_df['stocks_from_body']
reddit_df['mentioned_stocks'] = reddit_df['mentioned_stocks'].apply(lambda x: sorted(set(x)) if isinstance(x, list) else [])

# Second pass without "$"
all_stocks = sorted(set(chain.from_iterable(reddit_df['mentioned_stocks'])))
stock_pattern = r'\b(' + '|'.join(map(re.escape, all_stocks)) + r')\b'

print("\nLooking for previously extracted stock mentions in text without '$'.")
reddit_df['stocks_without_dollar_title'] = reddit_df['title'].str.extractall(stock_pattern).groupby(level=0).agg(list).reindex(reddit_df.index, fill_value=[])
reddit_df['stocks_without_dollar_body'] = reddit_df['body'].str.extractall(stock_pattern).groupby(level=0).agg(list).reindex(reddit_df.index, fill_value=[])

reddit_df['mentioned_stocks'] = reddit_df['mentioned_stocks'] + reddit_df['stocks_without_dollar_title'] + reddit_df['stocks_without_dollar_body']
reddit_df['mentioned_stocks'] = reddit_df['mentioned_stocks'].apply(lambda x: sorted(set(x)) if isinstance(x, list) else [])

#######################################
# Cleaning Text for Further Processing
#######################################

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()
    return text

print("\nCleaning title and body text.")
reddit_df['clean_title'] = reddit_df['title'].apply(clean_text)
reddit_df['clean_body'] = reddit_df['body'].apply(clean_text)

#######################################
# Save Excluded Rows Before Filtering
#######################################

rows_before_filtering = len(reddit_df)
print(f"\nNumber of rows before filtering: {rows_before_filtering}")

# Identify rows without stock mentions
excluded_df = reddit_df[reddit_df['mentioned_stocks'].apply(len) == 0]

# Save excluded rows (already cleaned)
excluded_file_path = "data/excluded_wallstreetbets_data.csv"
excluded_df.to_csv(excluded_file_path, index=False)
print(f"Excluded rows saved to '{excluded_file_path}' with {len(excluded_df)} rows.")

#######################################
# Filter Out Rows Without Stock Mentions
#######################################

print("\nExcluding posts without mention of a stock.")
reddit_df = reddit_df[reddit_df['mentioned_stocks'].apply(len) > 0]

rows_after_filtering = len(reddit_df)
print(f"Number of rows after filtering: {rows_after_filtering}")

# Reset index
reddit_df.reset_index(drop=True, inplace=True)

#######################################
# Counting mentioned stocks
#######################################

mention_number = 20
print(f"\nCounting the top {mention_number} most mentioned stocks.")

stock_counts = Counter(stock for stocks in reddit_df['mentioned_stocks'] for stock in stocks)
stock_counts_df = pd.DataFrame(stock_counts.items(), columns=['Stock', 'Mentions'])
stock_counts_df = stock_counts_df.sort_values(by='Mentions', ascending=False)
top_stock_counts_df = stock_counts_df.head(mention_number)

plt.figure(figsize=(12, 6))
plt.bar(top_stock_counts_df['Stock'], top_stock_counts_df['Mentions'], color='blue', alpha=0.7)
plt.title('Top 20 Most Mentioned Stocks')
plt.xlabel('Stock')
plt.ylabel('Number of Mentions')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

output_path = "results/top_mentioned_stocks.png"
plt.savefig(output_path)
print(f"A plot of the top {mention_number} most mentioned stocks has been saved to {output_path}.")
plt.close()

#######################################
# Save Cleaned Data
#######################################

cleaned_file_path = "data/cleaned_wallstreetbets_data.csv"
reddit_df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}.")


#######################################
# Analyze Length of Titles and Bodies
#######################################

# Calculate the length of each title and body in terms of word count
print("\nCalculating the length of each title and body in terms of word count.")
title_length = [len(word_tokenize(text)) for text in reddit_df['clean_title']]
body_length = [len(word_tokenize(text)) for text in reddit_df['clean_body']]

# Plot histograms for the length of titles and bodies
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(title_length, bins=50, kde=True, ax=axis1, color='blue')
sns.histplot(body_length, bins=40, kde=True, ax=axis2, color='green')

axis1.set_title("Distribution of Title Lengths")
axis1.set_xlabel("Length of Title")
axis1.set_ylabel("Frequency")

axis2.set_title("Distribution of Body Lengths")
axis2.set_xlabel("Length of Body")
axis2.set_ylabel("Frequency")

plt.tight_layout()

# Save the plot as a PNG file
output_path = "results/length_of_titles_and_bodies.png"
plt.savefig(output_path)
print(f"A plot of sentence lengths has been saved to {output_path}.")
plt.close()

#######################################
# Generate Word Clouds for Title and Body
#######################################

# Helper function to generate word cloud string
def generate_word_cloud_string(texts, stopwords):
    word_tokens = [word_tokenize(text) for text in texts]
    word_cloud_string = ""
    for word_list in word_tokens:
        for word in word_list:
            if word.lower() not in stopwords:  # Exclude stopwords
                word_cloud_string += word + " "
    return word_cloud_string

# Set of stopwords
description_stopwords = set(STOPWORDS)

# Generate Word Cloud for Titles
print("\nGenerating Word Cloud for titles")
title_word_cloud_string = generate_word_cloud_string(reddit_df['clean_title'], description_stopwords)
title_word_cloud = WordCloud(
    background_color='white',
    stopwords=description_stopwords,
    width=3000,  # Higher resolution width
    height=2000,  # Higher resolution height
    max_words=300,  # Include up to 300 words
    colormap='viridis',  # More visually appealing color map
    contour_width=1,  # Add contour around words
    contour_color='black'
).generate(title_word_cloud_string)

# Plot Word Cloud for Titles
plt.figure(figsize=(15, 10))  # Larger figure size for better display
plt.imshow(title_word_cloud, interpolation='bilinear')
plt.title("Word Cloud for Post Titles", fontsize=24)
plt.axis('off')
# Save the cloud as a PNG file
output_path = "results/wordcloud_titles.png"
plt.savefig(output_path)
print(f"A wordcloud of the titles has been saved to {output_path}.")
plt.close()

# Generate Word Cloud for Bodies
print("\nGenerating Word Cloud for bodies")
body_word_cloud_string = generate_word_cloud_string(reddit_df['clean_body'], description_stopwords)
body_word_cloud = WordCloud(
    background_color='white',
    stopwords=description_stopwords,
    width=3000,
    height=2000,
    max_words=300,
    colormap='plasma',
    contour_width=1,
    contour_color='black'
).generate(body_word_cloud_string)

# Plot Word Cloud for Bodies
plt.figure(figsize=(15, 10))  # Larger figure size for better display
plt.imshow(body_word_cloud, interpolation='bilinear')
plt.title("Word Cloud for Post Bodies", fontsize=24)
plt.axis('off')
# Save the cloud as a PNG file
output_path = "results/wordcloud_bodies.png"
plt.savefig(output_path)
print(f"A wordcloud of the bodies has been saved to {output_path}.")
plt.close()

print("\nText preprocessing has been completed.\n")