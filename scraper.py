import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import tempfile

# ---------------------- STEP 1: Scrape Job Listings ----------------------

def scrape_karkidi_jobs():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")

    driver = webdriver.Chrome(options=options)
    driver.get("https://www.karkidi.com/job-search")
    time.sleep(5)

    jobs = []

    job_cards = driver.find_elements("class name", "ads-details")

    print(f"Found {len(job_cards)} job cards")

    for card in job_cards:
        try:
            # Extract job title
            title_elem = card.find_element("tag name", "h4")
            title = title_elem.text.strip()

            # Extract link (optional, not needed now)
            link = card.find_element("tag name", "a").get_attribute("href")

            # Company name is within cmp-info or nearby
            try:
                company = card.find_element("class name", "cmp-info").text.strip()
            except:
                company = ""

            jobs.append({
                "title": title,
                "company": company,
                "skills": ""  # skills not visible in summary view
            })

        except Exception as e:
            print("Error parsing a job card:", e)
            continue

    driver.quit()
    return pd.DataFrame(jobs)

# ---------------------- STEP 2: Preprocess Skills ----------------------

def preprocess_skills(df):

    df["skills"] = df["title"].str.lower()
    return df

# ---------------------- STEP 3: Vectorize and Cluster ----------------------

def cluster_jobs(df, num_clusters=5):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(df["skills"])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    return df, kmeans, vectorizer

# ---------------------- STEP 4: Save Model ----------------------

def save_model(kmeans, vectorizer):
    joblib.dump(kmeans, "karkidi_kmeans_model.pkl")
    joblib.dump(vectorizer, "karkidi_vectorizer.pkl")

# ---------------------- MAIN PIPELINE ----------------------

def main():
    print("Scraping job listings...")
    df = scrape_karkidi_jobs()

    print("Preprocessing data...")
    df = preprocess_skills(df)

    print("Clustering jobs based on required skills...")
    df, kmeans, vectorizer = cluster_jobs(df, num_clusters=5)

    print("Saving model and vectorizer...")
    save_model(kmeans, vectorizer)

    print("Clustered Data Sample:")
    print(df.head())

    df.to_csv("karkidi_clustered_jobs.csv", index=False)

if __name__ == "__main__":
    main()
