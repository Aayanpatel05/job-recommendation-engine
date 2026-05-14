from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.pipeline.scheduler import start_scheduler
from src.pipeline.ingest_jobs import run_ingestion
from src.core.recommendation_engine import RecommendationEngine
from src.core.resume_parser import (
    clean_resume_text,
    extract_text_from_resume
)

import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield


# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(
    title="Job Recommendation API",
    lifespan=lifespan
)


# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Load Recommendation Engine
# -----------------------
try:
    engine = RecommendationEngine(
        model_name="all-MiniLM-L6-v2",
        index_path="data/processed/faiss_index.bin",
        jobs_path="data/processed/jobs_live.csv"
    )

    logger.info("Recommendation engine loaded successfully.")

except Exception as e:
    logger.error(f"Error loading RecommendationEngine: {e}")
    engine = None


# -----------------------
# Health Check
# -----------------------
@app.get("/")
def home():
    return {
        "message": "Job Recommendation API running"
    }


# -----------------------
# Recommend Jobs Endpoint
# -----------------------
@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    top_k: int = 10,
    preferred_location: str = None,
    experience_level: str = None
):

    global engine

    tmp_file_path = None

    try:

        # -----------------------
        # Save uploaded resume
        # -----------------------
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename)[1]
        ) as tmp_file:

            content = await file.read()
            tmp_file.write(content)

            tmp_file_path = tmp_file.name

        logger.info("Resume uploaded successfully.")

        # -----------------------
        # RUN INGESTION USING USER RESUME
        # -----------------------
        run_ingestion(
            resume_path=tmp_file_path,
            jobs_path="data/processed/jobs_live.csv",
            index_path="data/processed/faiss_index.bin"
        )

        logger.info("Job ingestion completed.")

        # -----------------------
        # Reload engine with NEW jobs
        # -----------------------
        engine = RecommendationEngine(
            model_name="all-MiniLM-L6-v2",
            index_path="data/processed/faiss_index.bin",
            jobs_path="data/processed/jobs_live.csv"
        )

        logger.info("Recommendation engine reloaded.")

        # -----------------------
        # Extract resume text
        # -----------------------
        resume_text = extract_text_from_resume(tmp_file_path)
        resume_text = clean_resume_text(resume_text)

        logger.info("Resume successfully extracted.")

        # -----------------------
        # Generate recommendations
        # -----------------------
        recommendations = engine.search_jobs(
            resume_text=resume_text,
            k=top_k,
            preferred_location=preferred_location,
            resume_experience=experience_level
        )

        # -----------------------
        # Empty result handling
        # -----------------------
        if recommendations is None or recommendations.empty:
            return {
                "source": "faiss_only",
                "recommendations": []
            }

        # -----------------------
        # Return results
        # -----------------------
        return {
            "source": "faiss_only",
            "recommendations": recommendations.to_dict(orient="records")
        }

    except Exception as e:

        logger.error(f"ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Cleanup temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# -----------------------
# Similar Jobs Endpoint
# -----------------------
@app.get("/similar_jobs/{job_id}")
def similar_jobs(job_id: int, top_k: int = 10):

    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine not available."
        )

    try:

        recommendations = engine.recommend_similar_jobs(
            job_id,
            k=top_k
        )

        return {
            "recommendations": recommendations.to_dict(orient="records")
        }

    except Exception as e:

        logger.error(f"ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )