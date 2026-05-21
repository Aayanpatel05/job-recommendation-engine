import requests


APP_ID = "c34d3b06"
APP_KEY = "3ddfce9885a375f118d3efe96d03582c"

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def fetch_jobs(
    query,
    location="us",
    results_per_page=50,
    max_pages=3,
    remote_only=False
):
    all_jobs = []
    seen_ids = set()

    for page in range(1, max_pages + 1):

        url = f"{BASE_URL}/{location}/search/{page}"

        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": results_per_page,
            "what": query,
            "content-type": "application/json",
        }

        response = requests.get(url, params=params)

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            break

        for job in results:

            job_id = job.get("id")

            if not job_id or job_id in seen_ids:
                continue

            seen_ids.add(job_id)

            title = job.get("title", "")
            description = job.get("description", "")
            location_name = job.get("location", {}).get("display_name", "")
            company = job.get("company", {}).get("display_name", "")
            redirect_url = job.get("redirect_url", "")

            full_text = f"""
            {title}

            {description}
            """

            # Optional remote filtering
            if remote_only:
                combined = f"{title} {description} {location_name}".lower()

                if "remote" not in combined:
                    continue

            all_jobs.append({
                "job_id": job_id,
                "title": title,
                "description": description,
                "full_text": full_text,
                "location": location_name,
                "company": company,
                "url": redirect_url,
            })

    return all_jobs