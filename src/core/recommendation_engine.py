import pandas as pd
import numpy as np
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
        job_descriptions = jobs_df["description"].fillna("").tolist()

        job_embeddings = self.model.encode(job_descriptions)
        resume_embedding = self.model.encode([resume_text])

        import numpy as np
        similarities = np.dot(job_embeddings, resume_embedding.T).flatten()

        jobs_df["similarity"] = similarities

        return jobs_df.sort_values(by="similarity", ascending=False).head(k)