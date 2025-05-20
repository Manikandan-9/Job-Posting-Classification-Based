# Job-Posting-Classification-Based

This code does:
1. Web Scraper:
Fetches job listings from karkidi.com.

2. Data Processing:
Uses job titles as proxy for required skills (smart workaround for missing detail page scraping).

3. Unsupervised Classification:
Clusters jobs into 5 groups using TF-IDF + KMeans.

4. Model Persistence:
Saves trained model and vectorizer with joblib

5. Saves scraped data into a csv file
