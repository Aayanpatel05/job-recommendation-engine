import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def detect_experience_level(title, description=""):
    """
    Determine experience level primarily from the job title.
    The title is much more reliable than searching the entire description.
    """

    title_lower = title.lower().strip()

    # Internship / co-op
    if re.search(r"\b(intern|internship|co[- ]?op)\b", title_lower):
        return "Internship"

    # Director / executive
    if re.search(
        r"\b(director|vice president|vp|chief|executive)\b",
        title_lower
    ):
        return "Executive"

    # Senior / lead / principal
    if re.search(
        r"\b(senior|sr\.?|lead|principal|staff)\b",
        title_lower
    ):
        return "Mid-Senior level"

    # Entry / junior
    if re.search(
        r"\b(entry[- ]level|junior|jr\.?)\b",
        title_lower
    ):
        return "Entry level"

    # Associate
    if re.search(r"\bassociate\b", title_lower):
        return "Associate"

    # Fallback: only inspect the description for very explicit phrases
    description_lower = description.lower()

    if re.search(
        r"\b(internship position|intern position|summer internship)\b",
        description_lower
    ):
        return "Internship"

    if re.search(
        r"\b(entry[- ]level position|entry[- ]level role)\b",
        description_lower
    ):
        return "Entry level"

    return "Unknown"


def fetch_jobs(
    query,
    location=None,
    results_per_page=50,
    max_pages=3,
    remote_only=False,
    experience_level=None
):
    """
    Fetch jobs from Adzuna.

    query:
        Job role / skills being searched.

    location:
        Preferred city/region.

    experience_level:
        Internship, Entry level, Associate, Mid-Senior level,
        Director, or Executive.
    """

    if not APP_ID or not APP_KEY:
        raise ValueError(
            "Adzuna credentials are missing. "
            "Set ADZUNA_APP_ID and ADZUNA_APP_KEY in .env"
        )

    all_jobs = []
    seen_ids = set()

    for page in range(1, max_pages + 1):

        url = f"{BASE_URL}/us/search/{page}"

        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": results_per_page,
            "what": query,
            "content-type": "application/json",
        }

        # Let Adzuna handle the initial location search
        if location:
            params["where"] = location

        try:
            response = requests.get(
                url,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

        except requests.RequestException as e:
            print(f"Failed fetching jobs for query '{query}': {e}")
            continue

        results = data.get("results", [])

        if not results:
            break

        for job in results:

            job_id = job.get("id")

            if not job_id or job_id in seen_ids:
                continue

            seen_ids.add(job_id)

            title = job.get("title", "").strip()
            description = job.get("description", "").strip()

            location_data = job.get("location", {})
            location_name = location_data.get(
                "display_name", ""
            ).strip()

            company_data = job.get("company", {})
            company = company_data.get(
                "display_name", ""
            ).strip()

            redirect_url = job.get("redirect_url", "")

            detected_level = detect_experience_level(
                title,
                description
            )

            # Remote detection
            combined_text = (
                f"{title} {description} {location_name}"
            ).lower()

            is_remote = any(
                phrase in combined_text
                for phrase in [
                    "remote",
                    "work from home",
                    "fully remote",
                    "remote position"
                ]
            )

            if remote_only and not is_remote:
                continue

            # Do NOT remove unknown experience jobs here.
            # Let the recommendation engine score them.
            if experience_level:
                if (
                    detected_level != experience_level
                    and detected_level != "Unknown"
                ):
                    # Keep remote/unknown jobs out only when
                    # the detected level directly conflicts.
                    continue

            full_text = f"""
            {title}
            {company}
            {location_name}
            {description}
            """

            all_jobs.append({
                "job_id": job_id,
                "title": title,
                "description": description,
                "full_text": full_text,
                "location": location_name,
                "company": company,
                "experience_level": detected_level,
                "url": redirect_url,
                "is_remote": is_remote
            })

    return all_jobs