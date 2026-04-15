import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
class RecommendationEngine:
    def __init__(self,
                 model_name='all-MiniLM-L6-v2',
                 index_path='data/processed/faiss_index.bin',
                 jobs_path='data/processed/jobs_preprocessed.csv'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.jobs = pd.read_csv(jobs_path)

        assert self.index.ntotal == len(self.jobs), "Index and jobs count mismatch"

    def encode_resume(self, resume_text: str) -> np.ndarray:
        embedding = self.model.encode([resume_text])
        embedding = np.array(embedding).astype('float32')

        faiss.normalize_L2(embedding)
        return embedding
    
    def search_jobs(self, resume_text: str, k: int = 10 ):
        resume_embedding = self.encode_resume(resume_text)
        scores, indices = self.index.search(resume_embedding, k)
        results = self.jobs.iloc[indices[0]].copy()

        results['similarity'] = scores[0]
        return results[['job_id','title','similarity','company_id','location']]
    
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
        return similar_jobs[['job_id','title','similarity','company_id','location']]
    
    def search_jobs_from_df(self, resume_text, jobs_df, k=10):
        jobs_df = jobs_df.copy()

        job_descriptions = jobs_df["description"].fillna("").tolist()

        job_embeddings = self.model.encode(job_descriptions)
        resume_embedding = self.model.encode([resume_text])

        import numpy as np
        semantic_scores = np.dot(job_embeddings, resume_embedding.T).flatten()

        resume_lower = resume_text.lower()

        def title_score(title):
            if not isinstance(title, str):
                return 0
            title = title.lower()
            if title in resume_lower:
                return 1.0
            return sum(word in resume_lower for word in title.split()) / len(title.split())

        jobs_df["title_score"] = jobs_df["title"].apply(title_score)

        # IMPORTANT: combine scores
        jobs_df["final_score"] = (
            0.7 * semantic_scores +
            0.3 * jobs_df["title_score"].values
        )

        results = jobs_df.sort_values("final_score", ascending=False).head(k)

        return results[["job_id", "title", "final_score", "company", "location"]]