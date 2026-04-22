import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.job_fetcher import fetch_jobs
from src.core.resume_parser import add_description_chunks_to_skills_desc

MODEL_NAME = "all-MiniLM-L6-v2"
JOBS_PATH = "data/processed/jobs_live.csv"
INDEX_PATH = "data/processed/faiss_index.bin"

def run_ingestion():
    print("Starting job ingestion...")
    jobs = fetch_jobs(query="data scientist") #update for a greater query later
    df = pd.DataFrame(jobs)

    if df.empty:
        print("No jobs fetched")
        return
    
    print(f"Fetched {len(df)} jobs")
    
    df = add_description_chunks_to_skills_desc(df)

    model = SentenceTransformer(MODEL_NAME)

    descriptions = df["description"].fillna("").tolist()
    embeddings = model.encode(descriptions)

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    df.to_csv(JOBS_PATH, index=False)

    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()