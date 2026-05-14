import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RecommendationEngine:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        index_path='data/processed/faiss_index.bin',
        jobs_path='data/processed/jobs_preprocessed.csv'
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.jobs = pd.read_csv(jobs_path)

        assert self.index.ntotal == len(self.jobs), "Index and jobs count mismatch"

    # -----------------------
    # Resume encoding
    # -----------------------
    def encode_resume(self, resume_text: str) -> np.ndarray:
        embedding = self.model.encode([resume_text])
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding)
        return embedding

    # -----------------------
    # Experience scoring (kept as-is)
    # -----------------------
    def experience_score(self, job_level, resume_level):
        if resume_level is None or job_level is None:
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
        resume_score = hierarchy.get(resume_level, 0)

        diff = abs(job_score - resume_score)

        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.6
        elif diff == 2:
            return 0.3
        else:
            return 0.0

    # -----------------------
    # Main search function (UPDATED)
    # -----------------------
    def search_jobs(
        self,
        resume_text,
        k=10,
        preferred_location=None,
        resume_experience=None
    ):

        # Step 1: FAISS search
        embedding = self.model.encode([resume_text]).astype("float32")
        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, k)

        matched_jobs = self.jobs.iloc[indices[0]].copy()
        matched_jobs["similarity"] = scores[0]

        # -----------------------
        # Step 2: LOCATION BOOST (NOT FILTER)
        # -----------------------
        def location_boost(loc):
            if not preferred_location:
                return 0.0
            if pd.isna(loc):
                return 0.0
            if preferred_location.lower() in str(loc).lower():
                return 0.25
            return 0.0

        matched_jobs["location_boost"] = matched_jobs["location"].apply(location_boost)

        # -----------------------
        # Step 3: EXPERIENCE BOOST (NOT FILTER)
        # -----------------------
        def exp_boost(job_exp):
            return self.experience_score(job_exp, resume_experience) * 0.3

        matched_jobs["experience_boost"] = matched_jobs["experience_level"].apply(exp_boost)

        # -----------------------
        # Step 4: FINAL SCORE (HYBRID RANKING)
        # -----------------------
        matched_jobs["final_score"] = (
            matched_jobs["similarity"]
            + matched_jobs["location_boost"]
            + matched_jobs["experience_boost"]
        )

        matched_jobs = matched_jobs.sort_values(
            by="final_score",
            ascending=False
        )

        # -----------------------
        # Step 5: return clean output
        # -----------------------
        expected_columns = [
            "job_id",
            "title",
            "location",
            "similarity",
            "experience_level",
            "final_score"
        ]

        existing_columns = [
            col for col in expected_columns
            if col in matched_jobs.columns
        ]

        return matched_jobs[existing_columns]

    # -----------------------
    # Similar jobs (unchanged)
    # -----------------------
    def recommend_similar_jobs(self, job_id: int, k: int = 10):

        job_index_list = self.jobs.index[self.jobs['job_id'] == job_id].tolist()
        if not job_index_list:
            raise ValueError(f"Job ID {job_id} not found.")

        job_index = job_index_list[0]
        job_embedding = self.index.reconstruct(job_index).reshape(1, -1).astype('float32')

        faiss.normalize_L2(job_embedding)

        scores, indices = self.index.search(job_embedding, k + 1)
        similar_jobs = self.jobs.iloc[indices[0][1:]].copy()

        similar_jobs['similarity'] = scores[0][1:]

        return similar_jobs[
            ['job_id', 'title', 'similarity', 'company_id', 'location']
        ]