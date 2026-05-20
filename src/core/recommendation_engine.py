import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RecommendationEngine:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # -----------------------
    # Encode text
    # -----------------------
    def encode_text(self, text_list):

        embeddings = self.model.encode(text_list)

        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        return embeddings

    # -----------------------
    # Experience boost
    # -----------------------
    def experience_boost(self, job_level, preferred_level):

        if not preferred_level or not job_level:
            return 0

        hierarchy = {
            "Internship": 0,
            "Entry level": 1,
            "Associate": 2,
            "Mid-Senior level": 3,
            "Director": 4,
            "Executive": 5
        }

        job_score = hierarchy.get(job_level, 0)
        preferred_score = hierarchy.get(preferred_level, 0)

        diff = abs(job_score - preferred_score)

        if diff == 0:
            return 0.15

        if diff == 1:
            return 0.08

        return 0

    # -----------------------
    # Location boost
    # -----------------------
    def location_boost(self, job_location, preferred_location):

        if not preferred_location or not job_location:
            return 0

        if preferred_location.lower() in job_location.lower():
            return 0.10

        return 0

    # -----------------------
    # Main recommendation logic
    # -----------------------
    def search_jobs(
        self,
        resume_text,
        jobs,
        k=10,
        preferred_location=None,
        preferred_experience=None
    ):

        if not jobs:
            return []

        jobs_df = pd.DataFrame(jobs)

        # -----------------------
        # Combine searchable text
        # -----------------------
        jobs_df["combined_text"] = (
            jobs_df["title"].fillna("") + " " +
            jobs_df["description"].fillna("")
        )

        # -----------------------
        # Encode jobs
        # -----------------------
        job_embeddings = self.encode_text(
            jobs_df["combined_text"].tolist()
        )

        # -----------------------
        # Temporary in-memory FAISS index
        # -----------------------
        dimension = job_embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(job_embeddings)

        # -----------------------
        # Encode resume
        # -----------------------
        resume_embedding = self.encode_text([resume_text])

        # -----------------------
        # Search
        # -----------------------
        scores, indices = index.search(
            resume_embedding,
            min(k * 5, len(jobs_df))
        )

        matched_jobs = jobs_df.iloc[indices[0]].copy()

        matched_jobs["similarity"] = scores[0]

        # -----------------------
        # Add boosters
        # -----------------------
        matched_jobs["boost"] = matched_jobs.apply(
            lambda row:
                self.location_boost(
                    row.get("location"),
                    preferred_location
                )
                +
                self.experience_boost(
                    row.get("experience_level"),
                    preferred_experience
                ),
            axis=1
        )

        matched_jobs["final_score"] = (
            matched_jobs["similarity"]
            + matched_jobs["boost"]
        )

        matched_jobs = matched_jobs.sort_values(
            by="final_score",
            ascending=False
        )

        return matched_jobs.head(k).to_dict(orient="records")