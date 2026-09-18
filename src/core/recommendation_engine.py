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

        embeddings = self.model.encode(
            text_list,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        return embeddings

    # -----------------------
    # Experience matching
    # -----------------------
    def experience_score(self, job_level, preferred_level):

        if not preferred_level or not job_level:
            return 0.0

        job_level = str(job_level).lower().strip()
        preferred_level = str(preferred_level).lower().strip()

        # Normalize different ways of writing the same level
        level_aliases = {
            "internship": "intern",
            "intern": "intern",

            "entry level": "entry",
            "entry-level": "entry",
            "entry": "entry",
            "new grad": "entry",

            "associate": "mid",
            "mid level": "mid",
            "mid-level": "mid",
            "mid": "mid",

            "mid-senior level": "senior",
            "senior": "senior",

            "director": "advanced",
            "executive": "advanced",
            "lead": "advanced",
            "principal": "advanced",
            "advanced": "advanced"
        }

        job_level = level_aliases.get(job_level, job_level)
        preferred_level = level_aliases.get(
            preferred_level,
            preferred_level
        )

        # Exact match
        if job_level == preferred_level:
            return 1.0

        # Interns and entry-level positions are closely related
        if preferred_level == "intern" and job_level == "entry":
            return 0.75

        if preferred_level == "entry" and job_level == "intern":
            return 0.75

        # Mid and senior are somewhat related
        if preferred_level == "mid" and job_level == "senior":
            return 0.50

        if preferred_level == "senior" and job_level == "mid":
            return 0.50

        # Everything else is a weak match
        return 0.0

    # -----------------------
    # Location matching
    # -----------------------
    def location_score(self, job_location, preferred_location):

        if not preferred_location or not job_location:
            return 0.0

        job_location = str(job_location).lower().strip()
        preferred_location = str(preferred_location).lower().strip()

        # Exact location match
        if preferred_location in job_location:
            return 1.0

        # Remote job
        remote_terms = [
            "remote",
            "work from home",
            "work-from-home",
            "fully remote"
        ]

        if any(term in job_location for term in remote_terms):
            return 0.75

        # Compare city/state components
        preferred_parts = [
            part.strip()
            for part in preferred_location.split(",")
        ]

        # City match
        if preferred_parts:
            city = preferred_parts[0]

            if city in job_location:
                return 0.90

        # State match
        if len(preferred_parts) >= 2:

            state = preferred_parts[-1]

            if state in job_location:
                return 0.65

        return 0.0

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

        # Encode resume
        embedding = self.model.encode(
            [resume_text],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        faiss.normalize_L2(embedding)

        # Use custom jobs dataset if provided
        search_jobs_df = (
            jobs
            if jobs is not None
            else self.jobs
        )

        if search_jobs_df is None or len(search_jobs_df) == 0:
            return pd.DataFrame()

        # -----------------------
        # Create job embeddings
        # -----------------------

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

        temp_index = faiss.IndexFlatIP(
            job_embeddings.shape[1]
        )

        temp_index.add(job_embeddings)

        # Retrieve a larger candidate pool
        candidate_count = min(
            max(k * 10, 50),
            len(search_jobs_df)
        )

        scores, indices = temp_index.search(
            embedding,
            candidate_count
        )

        # -----------------------
        # Build matched jobs
        # -----------------------

        valid_indices = indices[0][indices[0] >= 0]

        matched_jobs = search_jobs_df.iloc[
            valid_indices
        ].copy()

        matched_jobs["similarity"] = scores[0][
            :len(valid_indices)
        ]

        # -----------------------
        # Location scoring
        # -----------------------

        if preferred_location:

            matched_jobs["location_match"] = (
                matched_jobs["location"]
                .fillna("")
                .apply(
                    lambda x: self.location_score(
                        x,
                        preferred_location
                    )
                )
            )

        else:

            matched_jobs["location_match"] = 0.0

        # -----------------------
        # Experience scoring
        # -----------------------

        if (
            preferred_experience
            and "experience_level" in matched_jobs.columns
        ):

            matched_jobs["experience_match"] = (
                matched_jobs["experience_level"]
                .fillna("")
                .apply(
                    lambda x: self.experience_score(
                        x,
                        preferred_experience
                    )
                )
            )

        else:

            matched_jobs["experience_match"] = 0.0

        # -----------------------
        # Final recommendation score
        # -----------------------

        #
        # Semantic similarity:
        #     70%
        #
        # Location:
        #     20%
        #
        # Experience:
        #     10%
        #

        matched_jobs["final_score"] = (
            matched_jobs["similarity"] * 0.70
            + matched_jobs["location_match"] * 0.20
            + matched_jobs["experience_match"] * 0.10
        )

        # Convert to percentage for display
        matched_jobs["match_percentage"] = (
            matched_jobs["final_score"] * 100
        ).round(1)

        # -----------------------
        # Sort recommendations
        # -----------------------

        matched_jobs = matched_jobs.sort_values(
            by="final_score",
            ascending=False
        )

        # Remove duplicate job titles
        matched_jobs = matched_jobs.drop_duplicates(
            subset=["title"],
            keep="first"
        )

        # -----------------------
        # Return results
        # -----------------------

        expected_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "experience_level",
            "similarity",
            "location_match",
            "experience_match",
            "final_score",
            "match_percentage",
            "url"
        ]

        existing_columns = [
            col
            for col in expected_columns
            if col in matched_jobs.columns
        ]

        return matched_jobs.head(k)[existing_columns]
