from fastapi import FastAPI, UploadFile, File, HTTPException
from src.core.recommendation_engine import RecommendationEngine
from src.core.resume_parser import extract_text_from_resume, clean_resume_text
from src.core.job_fetcher import fetch_jobs
import tempfile
import os
import pandas as pd

app = FastAPI(title='Job Recommendation API')

try:
    engine = RecommendationEngine(
        model_name='all-MiniLM-L6-v2',
        index_path='data/processed/faiss_index.bin',
        jobs_path='data/processed/jobs_preprocessed.csv'
    )
except Exception as e:
    print("Error loading RecommendationEngine:", e)
    engine = None

@app.get("/")
def home():
    return {"message:" "Job Recommendation API is running. Use /recommend endpoint to get job recommendations."}

@app.post("/recommend")
async def recommend(file: UploadFile = File(...), top_k: int = 10):
    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine not available.")
    
    tmp_file_path = None
    try:
        # Save uploaded file as bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract text from PDF/DOCX safely
        resume_text = extract_text_from_resume(tmp_file_path)  # make sure this reads bytes, not text
        resume_text = clean_resume_text(resume_text)
        
        jobs = fetch_jobs(query="data scientist")
        jobs_df = pd.DataFrame(jobs)

        if jobs_df.empty:
            raise HTTPException(status_code=500,detail="No jobs fetched from API.")
        
        recommendations = engine.search_jobs_from_df(
            resume_text,
            jobs_df,
            k=top_k
        )
        return {
            "source": "live_api",
            "recommendations": recommendations.to_dict(orient='records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always remove temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.get("/similar_jobs/{job_id}")
def similar_jobs(job_id: int, top_k: int = 10):
    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine not available.")
    
    try:
        recommendations = engine.recommend_similar_jobs(job_id, k=top_k)
        return {"recommendations": recommendations.to_dict(orient='records')}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))