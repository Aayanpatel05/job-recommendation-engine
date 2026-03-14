import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
class RecommendationEngine:
    def __init__(self,
                 model_name='all-MiniLM-L6-v2',
                 index_path='data/processed/faiss_index.bin',
                 jobs_path='data/processed/jobs.csv'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.jobs = pd.read_csv(jobs_path)

    def encode_resume(self, resume_text):
        return self.model.encode([resume_text])
    
    def search_jobs(self, resume_text: str, k: int = 10 ):
        resume_embedding = self.encode_resume(resume_text)
        distances, indices = self.index.search(resume_embedding, k)
        results = self.jobs.iloc[indices[0]].copy()

        results['similarity'] = 1 - distances[0]
        return results[['job_id','title','similarity','company_id','location']]
    
    def recommend_similar_jobs(self, job_id: int, k: int = 10):
        job_index = self.jobs.index[self.jobs['job_id'] == job_id].tolist()
        if not job_index:
            raise ValueError(f"Job ID {job_id} not found.")
        
        job_embedding = self.index.reconstruct(job_index[0]).reshape(1, -1)
        distances, indices = self.index.search(job_embedding, k + 1)
        similar_jobs = self.jobs.iloc[indices[0][1:]].copy() 

        similar_jobs['similarity'] = 1 - distances[0][1:]
        return similar_jobs[['job_id','title','similarity','company_id','location']]