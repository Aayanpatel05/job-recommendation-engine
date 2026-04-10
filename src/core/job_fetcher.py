import requests

APP_ID="c34d3b06"
APP_KEY="3ddfce9885a375f118d3efe96d03582c"

def fetch_jobs(query="data scientist",location="us",results_per_page=20):
    url=f"https://api.adzuna.com/v1/api/jobs/{location}/search/1"

    params={
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": query,
    }

    response = requests.get(url,params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
    
    data = response.json()

    jobs=[]
    for job in data["results"]:
        jobs.append({
            "job_id": job.get("id"),
            "title": job.get("title"),
            "description": job.get("description"),
            "location": job.get("location", {}).get("display_name"),
            "company": job.get("company", {}).get("display_name")
        })
    return jobs