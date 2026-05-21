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
        jobs=None,
        k=10,
        preferred_location=None,
        preferred_experience=None
    ):

        embedding = self.model.encode([resume_text]).astype("float32")
        faiss.normalize_L2(embedding)

        # Use custom jobs dataset if provided
        search_jobs_df = jobs if jobs is not None else self.jobs

        # Build temporary embeddings/index if custom jobs passed
        if jobs is not None:

            combined_text = (
                search_jobs_df["title"].fillna("") + " " +
                search_jobs_df["description"].fillna("")
            ).tolist()

            job_embeddings = self.model.encode(
                combined_text,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

            faiss.normalize_L2(job_embeddings)

            temp_index = faiss.IndexFlatIP(job_embeddings.shape[1])
            temp_index.add(job_embeddings)

            scores, indices = temp_index.search(
                embedding,
                min(k * 5, len(search_jobs_df))
            )

        else:

            scores, indices = self.index.search(
                embedding,
                min(k * 5, len(search_jobs_df))
            )

        matched_jobs = search_jobs_df.iloc[indices[0]].copy()

        matched_jobs["similarity"] = scores[0]

        # -----------------------
        # Score Boosting
        # -----------------------

        final_scores = matched_jobs["similarity"].copy()

        # Location boost
        if preferred_location:
            location_match = matched_jobs["location"].fillna("").str.contains(
                preferred_location,
                case=False
            )

            final_scores += location_match.astype(float) * 0.15

        # Experience boost
        if preferred_experience and "experience_level" in matched_jobs.columns:

            exp_boosts = matched_jobs["experience_level"].apply(
                lambda x: self.experience_score(x, preferred_experience)
            )

            final_scores += exp_boosts * 0.10

        matched_jobs["final_score"] = final_scores

        matched_jobs = matched_jobs.sort_values(
            by="final_score",
            ascending=False
        )

        matched_jobs = matched_jobs.drop_duplicates(
            subset=["title"],
            keep="first"
        )

        expected_columns = [
            "job_id",
            "title",
            "location",
            "similarity",
            "final_score",
            "experience_level"
        ]

        existing_columns = [
            col for col in expected_columns
            if col in matched_jobs.columns
        ]

        return matched_jobs.head(k)[existing_columns]