import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.job_fetcher import fetch_jobs
from src.core.resume_parser import (
    add_description_chunks_to_skills_desc,
    extract_text_from_resume,
    clean_resume_text,
    extract_experience_level
)
from src.core.query_generator import generate_queries_from_resume

MODEL_NAME = "all-MiniLM-L6-v2"


def run_ingestion(
    resume_path,
    jobs_path="data/processed/jobs_live.csv",
    index_path="data/processed/faiss_index.bin"
):

    resume_text = extract_text_from_resume(resume_path)
    resume_text = clean_resume_text(resume_text)

    queries = generate_queries_from_resume(resume_text)

    print("Generated Queries:")
    print(queries)

    all_jobs = []

    print("Starting job ingestion...")

    for query in queries:

        for page in range(1, 4):

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

    # -----------------------
    # Remove duplicates
    # -----------------------

    df["title_company"] = (
        df["title"].astype(str).str.lower()
        + "_"
        + df["company"].astype(str).str.lower()
    )

    df = df.drop_duplicates(subset=["title_company"])
    df = df.drop_duplicates(subset=["job_id"])

    # -----------------------
    # Experience extraction
    # -----------------------

    df["experience_level"] = df.apply(
        lambda row: extract_experience_level(
            row["title"],
            row["description"],
            row.get("experience_level")
        ),
        axis=1
    )

    print(f"Fetched {len(df)} jobs")

    # -----------------------
    # Build skills description
    # -----------------------

    df = add_description_chunks_to_skills_desc(df)

    # -----------------------
    # Embeddings
    # -----------------------

    model = SentenceTransformer(MODEL_NAME)

    descriptions = (
        df["description"].fillna("")
        + " "
        + df["skills_desc"].fillna("")
    ).tolist()

    embeddings = model.encode(
        descriptions,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    # -----------------------
    # Build FAISS index
    # -----------------------

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # -----------------------
    # Save
    # -----------------------

    faiss.write_index(index, index_path)

    df.to_csv(jobs_path, index=False)

    print("Ingestion complete!")
    print(f"Saved {len(df)} jobs")