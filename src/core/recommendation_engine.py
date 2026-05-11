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
    # Experience scoring
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
    # Main search function
    # -----------------------
    def search_jobs(
        self,
        resume_text: str,
        k: int = 10,
        preferred_location: str = None,
        user_experience_level: str = None,
        resume_experience: str = None
    ):

        resume_embedding = self.encode_resume(resume_text)

        # FAISS retrieval
        scores, indices = self.index.search(resume_embedding, k * 10)
        candidates = self.jobs.iloc[indices[0]].copy()
        candidates["similarity"] = scores[0]

        # -----------------------
        # Location boost
        # -----------------------
        if preferred_location:

            preferred = preferred_location.lower()

            def location_boost(loc):
                if not isinstance(loc, str):
                    return 0

                loc = loc.lower()

                if preferred in loc:
                    return 0.15

                if "remote" in loc:
                    return 0.08

                return 0

            candidates["similarity"] += candidates["location"].apply(location_boost)

        # -----------------------
        # EXPERIENCE FILTER + SCORING
        # -----------------------
        effective_experience = user_experience_level or resume_experience

        hierarchy = {
            "Internship": 0,
            "Entry level": 1,
            "Associate": 2,
            "Mid-Senior level": 3,
            "Director": 4,
            "Executive": 5
        }

        if effective_experience:

            resume_rank = hierarchy.get(effective_experience, 1)

            # HARD FILTER
            def is_valid_experience(job_level):
                if pd.isna(job_level):
                    return False

                job_rank = hierarchy.get(job_level, 1)

                # allow ±1 band
                return abs(job_rank - resume_rank) <= 1

            candidates = candidates[
                candidates["experience_level"].apply(is_valid_experience)
            ]

            # SOFT SCORING
            def exp_score(row):
                return self.experience_score(
                    row.get("experience_level"),
                    effective_experience
                )

            candidates["experience_score"] = candidates.apply(exp_score, axis=1)

            # stronger weighting
            candidates["similarity"] += 0.25 * candidates["experience_score"]

        # -----------------------
        # Remove duplicate titles
        # -----------------------
        selected = []
        used_titles = set()

        for _, row in candidates.sort_values("similarity", ascending=False).iterrows():

            title = str(row["title"]).lower()

            normalized = (
                title.replace("senior", "")
                     .replace("sr", "")
                     .replace("junior", "")
                     .replace("jr", "")
                     .strip()
            )

            if any(normalized in t or t in normalized for t in used_titles):
                continue

            selected.append(row)
            used_titles.add(normalized)

            if len(selected) >= k:
                break

        results = pd.DataFrame(selected)

        return results[
            ["job_id", "title", "similarity", "location"]
        ]

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