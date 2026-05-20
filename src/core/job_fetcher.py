import requests

APP_ID = "c34d3b06"
APP_KEY = "3ddfce9885a375f118d3efe96d03582c"

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def fetch_jobs(
    query="jobs",
    location="us",
    results_per_page=50,
    page=1
):
    url = f"{BASE_URL}/{location}/search/{page}"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": query,
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    jobs = []

    for job in data.get("results", []):

        jobs.append({
            "job_id": job.get("id"),
            "title": job.get("title"),
            "description": job.get("description"),
            "location": job.get("location", {}).get("display_name"),
            "company": job.get("company", {}).get("display_name"),
        })

    return jobs