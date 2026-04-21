from fastapi import FastAPI, UploadFile, File, HTTPException
from src.core.recommendation_engine import RecommendationEngine
from src.core.resume_parser import (
    extract_text_from_resume,
    clean_resume_text,
    add_description_chunks_to_skills_desc
)
from src.core.job_fetcher import fetch_jobs

import tempfile
import os
import pandas as pd
import logging

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Job Recommendation API")

# -----------------------
# Load engine
# -----------------------
try:
    engine = RecommendationEngine(
        model_name="all-MiniLM-L6-v2",
        index_path="data/processed/faiss_index.bin",
        jobs_path="data/processed/jobs_preprocessed.csv"
    )
except Exception as e:
    logger.error(f"Error loading RecommendationEngine: {e}")
    engine = None


# -----------------------
# Health check
# -----------------------
@app.get("/")
def home():
    return {
        "message": "Job Recommendation API is running. Use /recommend endpoint."
    }


# -----------------------
# MAIN ENDPOINT
# -----------------------
@app.post("/recommend")
async def recommend(file: UploadFile = File(...), top_k: int = 10):

    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine not available.")

    tmp_file_path = None

    try:
        # Save uploaded resume
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Extract resume text
        resume_text = extract_text_from_resume(tmp_file_path)
        resume_text = clean_resume_text(resume_text)

        # Fetch jobs from API
        jobs = fetch_jobs(query="data scientist")
        jobs_df = pd.DataFrame(jobs)

        if jobs_df.empty:
            raise HTTPException(status_code=500, detail="No jobs fetched from API.")

        # Ensure required columns exist
        if "description" not in jobs_df.columns:
            raise HTTPException(status_code=500, detail="API response missing 'description' field.")

        # Create skills column
        jobs_df = add_description_chunks_to_skills_desc(jobs_df)

        # Run recommendation
        recommendations = engine.search_jobs_from_df(
            resume_text,
            jobs_df,
            k=top_k
        )

        if recommendations is None or recommendations.empty:
            raise HTTPException(status_code=500, detail="No recommendations generated.")

        return {
            "source": "live_api",
            "recommendations": recommendations.to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# -----------------------
# Similar jobs endpoint
# -----------------------
@app.get("/similar_jobs/{job_id}")
def similar_jobs(job_id: int, top_k: int = 10):

    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine not available.")

    try:
        recommendations = engine.recommend_similar_jobs(job_id, k=top_k)
        return {"recommendations": recommendations.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))