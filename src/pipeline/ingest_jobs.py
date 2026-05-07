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
    QUERIES = [
    "jobs",
    "remote",
    "intern",
    "engineer",
    "analyst",
    "manager",
    "developer",
    "assistant",
    "specialist",
    "coordinator" 
    ]
    all_jobs=[]
    print("Starting job ingestion...")
    for query in QUERIES:
        for page in range(1, 6):
            jobs = fetch_jobs(
                query=query,
                page=page,
                results_per_page=50
            )

            all_jobs.extend(jobs)
    
    df = pd.DataFrame(all_jobs)

    if df.empty:
        print("No jobs fetched")
        return
    
    df= df.drop_duplicates(subset=['job_id'])
    
    print(f"Fetched {len(df)} jobs")
    
    df = add_description_chunks_to_skills_desc(df)

    model = SentenceTransformer(MODEL_NAME)

    descriptions = (
        df["description"].fillna("") +
        " " +
        df["skills_desc"].fillna("")
    ).tolist()

    embeddings = model.encode(
        descriptions,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)


    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)


    faiss.write_index(index, INDEX_PATH)

    df.to_csv(JOBS_PATH, index=False)

    print("Ingestion complete!")
    print(f"Saved {len(df)} jobs")
    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved jobs CSV to: {JOBS_PATH}")

if __name__ == "__main__":
    run_ingestion()