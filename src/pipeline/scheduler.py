from apscheduler.schedulers.background import BackgroundScheduler
from src.pipeline.ingest_jobs import run_ingestion

def start_scheduler():
    scheduler = BackgroundScheduler()

    scheduler.add_job(run_ingestion, "interval", hours = 6)

    scheduler.start()