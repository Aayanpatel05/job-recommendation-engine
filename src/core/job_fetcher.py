import requests


APP_ID = "YOUR_ADZUNA_APP_ID"
APP_KEY = "YOUR_ADZUNA_APP_KEY"

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def detect_experience_level(title, description):
    """
    Detect the likely experience level required by a job posting.
    """

    text = f"{title} {description}".lower()

    # Internship / Co-op
    if any(term in text for term in [
        "intern",
        "internship",
        "co-op",
        "coop",
        "co op"
    ]):
        return "intern"

    # Entry level / New Grad
    if any(term in text for term in [
        "entry level",
        "entry-level",
        "junior",
        "new grad",
        "new graduate",
        "graduate program",
        "early career"
    ]):
        return "entry"

    # Senior
    if any(term in text for term in [
        "senior",
        "sr.",
        "sr ",
        "5+ years",
        "6+ years",
        "7+ years",
        "8+ years"
    ]):
        return "senior"

    # Lead / Principal / Staff
    if any(term in text for term in [
        "lead",
        "principal",
        "staff engineer",
        "staff scientist",
        "director"
    ]):
        return "advanced"

    # Default
    return "mid"


def fetch_jobs(
    query,
    location="us",
    results_per_page=50,
    max_pages=3,
    remote_only=False,
    desired_location=None,
    experience_level=None
):
    """
    Fetch jobs from the Adzuna API.

    Parameters:
        query: Job search query.
        location: Adzuna country code.
        results_per_page: Number of jobs returned per page.
        max_pages: Maximum number of pages to retrieve.
        remote_only: If True, only return jobs mentioning remote work.
        desired_location: User's desired location, e.g. "Atlanta, GA".
        experience_level: User's desired experience level, e.g. "intern".

    Returns:
        List of job dictionaries.
    """

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

        # Let Adzuna perform an initial location filter
        if desired_location:
            params["where"] = desired_location

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            break

        for job in results:

            job_id = job.get("id")

            # Skip jobs without IDs or duplicate jobs
            if not job_id or job_id in seen_ids:
                continue

            seen_ids.add(job_id)

            title = job.get("title", "")
            description = job.get("description", "")

            location_name = job.get(
                "location", {}
            ).get(
                "display_name", ""
            )

            company = job.get(
                "company", {}
            ).get(
                "display_name", ""
            )

            # Adzuna application/job listing URL
            redirect_url = job.get("redirect_url", "")

            # Detect experience requirement
            detected_experience = detect_experience_level(
                title,
                description
            )

            # Combine information used by the embedding model
            full_text = f"""
            {title}

            {company}

            {location_name}

            {description}
            """

            combined = (
                f"{title} "
                f"{description} "
                f"{location_name}"
            ).lower()

            # Remote filtering
            if remote_only:
                remote_terms = [
                    "remote",
                    "work from home",
                    "work-from-home",
                    "fully remote"
                ]

                if not any(term in combined for term in remote_terms):
                    continue

            # Optional experience filtering
            #
            # This does NOT affect semantic ranking yet.
            # It simply allows the caller to request jobs
            # matching a specific experience level.
            if experience_level:
                requested_level = experience_level.lower().strip()

                if requested_level not in [
                    "all",
                    "any",
                    ""
                ]:

                    # Allow closely related entry-level terms
                    if requested_level == "intern":
                        valid_levels = ["intern"]

                    elif requested_level in [
                        "entry",
                        "entry-level",
                        "entry level",
                        "new grad"
                    ]:
                        valid_levels = ["entry", "intern"]

                    elif requested_level == "senior":
                        valid_levels = ["senior"]

                    elif requested_level in [
                        "mid",
                        "mid-level",
                        "mid level"
                    ]:
                        valid_levels = ["mid"]

                    else:
                        valid_levels = [requested_level]

                    if detected_experience not in valid_levels:
                        continue

            all_jobs.append({
                "job_id": job_id,
                "title": title,
                "description": description,
                "full_text": full_text,
                "location": location_name,
                "company": company,

                # New structured field
                "experience_level": detected_experience,

                # Application/listing URL
                "url": redirect_url,
            })

    return all_jobs
