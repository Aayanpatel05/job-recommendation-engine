from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline.scheduler import start_scheduler
from src.core.job_fetcher import fetch_jobs
from src.core.query_generator import generate_queries_from_resume

from src.core.recommendation_engine import RecommendationEngine
from src.core.resume_parser import (
    extract_text_from_resume,
    clean_resume_text
)

import tempfile
import logging
import os
import pandas as pd

# -----------------------
# Logging
# -----------------------
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
# Load Recommendation Engine ONCE
# -----------------------
try:

    engine = RecommendationEngine(
        model_name="all-MiniLM-L6-v2"
    )

    logger.info("Recommendation engine loaded successfully.")

except Exception as e:

    logger.error(f"Failed to load RecommendationEngine: {e}")

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

    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine not available."
        )

    tmp_file_path = None

    try:

        # -----------------------
        # Save uploaded resume
        # -----------------------
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename)[1]
        ) as tmp_file:

            tmp_file.write(await file.read())

            tmp_file_path = tmp_file.name

        logger.info("Resume uploaded successfully.")

        # -----------------------
        # Extract resume text
        # -----------------------
        resume_text = extract_text_from_resume(tmp_file_path)

        resume_text = clean_resume_text(resume_text)

        logger.info("Resume extracted successfully.")

        # -----------------------
        # Generate queries
        # -----------------------
        queries = generate_queries_from_resume(
            resume_text,
            max_queries=5
        )

        logger.info(f"Generated queries: {queries}")

        # -----------------------
        # Fetch jobs
        # -----------------------
        all_jobs = []

        for query in queries:

            try:

                jobs = fetch_jobs(
                    query=query,
                    results_per_page=20
                )

                all_jobs.extend(jobs)

            except Exception as e:

                logger.warning(
                    f"Failed fetching jobs for query '{query}': {e}"
                )

        # Convert to dataframe AFTER collecting everything
        all_jobs = pd.DataFrame(all_jobs)

        logger.info(f"Fetched {len(all_jobs)} jobs.")

        # -----------------------
        # Generate recommendations
        # -----------------------
        recommendations = engine.search_jobs(
            resume_text=resume_text,
            jobs=all_jobs,
            k=top_k,
            preferred_location=preferred_location,
            preferred_experience=experience_level
        )

        return {
            "source": "dynamic_live_jobs",
            "total_jobs_fetched": len(all_jobs),
            "recommendations": recommendations.to_dict(orient="records")
        }

    except Exception as e:

        logger.error(f"ERROR: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # -----------------------
        # Cleanup temp file
        # -----------------------
        if tmp_file_path and os.path.exists(tmp_file_path):

            os.remove(tmp_file_path)


# -----------------------
# Similar Jobs Endpoint
# -----------------------
@app.get("/similar_jobs")
def similar_jobs():

    return {
        "message": "Similar jobs endpoint temporarily disabled."
    }