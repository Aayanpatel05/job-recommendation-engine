import re


DEFAULT_QUERIES = [
    "software engineering internship",
    "backend engineering internship",
    "data science internship",
    "machine learning internship",
    "data analyst internship"
]


def generate_queries_from_resume(
    resume_text,
    max_queries=5,
    experience_level=None
):
    """
    Generate role-based job-search queries from a resume.

    Instead of searching individual resume phrases such as
    'webhooks' or 'woocommerce', this identifies likely job
    categories based on the candidate's skills and experience.
    """

    text = resume_text.lower()

    queries = []

    # --------------------------------------------------
    # Experience term
    # --------------------------------------------------

    experience_terms = {
        "Internship": "internship",
        "Entry level": "entry level",
        "Associate": "associate",
        "Mid-Senior level": "senior",
        "Director": "director",
        "Executive": "executive"
    }

    experience_term = experience_terms.get(
        experience_level,
        ""
    )

    def add_query(role):
        if experience_term:
            query = f"{role} {experience_term}"
        else:
            query = role

        if query not in queries:
            queries.append(query)

    # --------------------------------------------------
    # Software Engineering
    # --------------------------------------------------

    software_keywords = [
        "java",
        "python",
        "javascript",
        "c++",
        "c#",
        "spring boot",
        "software engineering",
        "software development",
        "programming",
        "git",
        "github"
    ]

    if any(keyword in text for keyword in software_keywords):
        add_query("software engineering")

    # --------------------------------------------------
    # Backend Engineering
    # --------------------------------------------------

    backend_keywords = [
        "backend",
        "back-end",
        "spring boot",
        "rest api",
        "rest apis",
        "api",
        "webhook",
        "webhooks",
        "fastapi",
        "server",
        "database",
        "sql"
    ]

    if any(keyword in text for keyword in backend_keywords):
        add_query("backend engineering")

    # --------------------------------------------------
    # Data Science
    # --------------------------------------------------

    data_science_keywords = [
        "data science",
        "pandas",
        "numpy",
        "scikit-learn",
        "sklearn",
        "regression",
        "statistics",
        "statistical",
        "data analysis",
        "data analytics"
    ]

    if any(keyword in text for keyword in data_science_keywords):
        add_query("data science")

    # --------------------------------------------------
    # Machine Learning / AI
    # --------------------------------------------------

    ml_keywords = [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "pytorch",
        "tensorflow",
        "transformer",
        "bert",
        "llm",
        "natural language processing",
        "nlp",
        "computer vision",
        "mediapipe"
    ]

    if any(keyword in text for keyword in ml_keywords):
        add_query("machine learning")

    # --------------------------------------------------
    # Data Analyst
    # --------------------------------------------------

    analyst_keywords = [
        "data analyst",
        "data analysis",
        "analytics",
        "sql",
        "tableau",
        "power bi",
        "excel",
        "statistics",
        "regression"
    ]

    if any(keyword in text for keyword in analyst_keywords):
        add_query("data analyst")

    # --------------------------------------------------
    # Full Stack Development
    # --------------------------------------------------

    fullstack_keywords = [
        "full stack",
        "fullstack",
        "react",
        "frontend",
        "front-end",
        "javascript",
        "html",
        "css"
    ]

    if any(keyword in text for keyword in fullstack_keywords):
        add_query("full stack developer")

    # --------------------------------------------------
    # Cloud / Integration Engineering
    # --------------------------------------------------

    integration_keywords = [
        "api integration",
        "integration",
        "woocommerce",
        "webhooks",
        "rest api",
        "rest apis",
        "cloud",
        "aws",
        "azure",
        "gcp"
    ]

    if any(keyword in text for keyword in integration_keywords):
        add_query("software integration engineer")

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    unique_queries = []

    for query in queries:

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        if query not in unique_queries:
            unique_queries.append(query)

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not unique_queries:

        return DEFAULT_QUERIES[:max_queries]

    return unique_queries[:max_queries]