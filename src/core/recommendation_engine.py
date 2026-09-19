import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class RecommendationEngine:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # --------------------------------------------------
    # Text Embeddings
    # --------------------------------------------------

    def encode_text(self, text_list):

        embeddings = self.model.encode(
            text_list,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        return embeddings

    # --------------------------------------------------
    # Experience Score
    # --------------------------------------------------

    def experience_score(
        self,
        job_level,
        preferred_level
    ):

        if not preferred_level or not job_level:
            return 0.5

        if job_level == "Unknown":
            return 0.5

        hierarchy = {
            "Internship": 0,
            "Entry level": 1,
            "Associate": 2,
            "Mid-Senior level": 3,
            "Director": 4,
            "Executive": 5
        }

        job_score = hierarchy.get(job_level)
        preferred_score = hierarchy.get(preferred_level)

        if job_score is None or preferred_score is None:
            return 0.5

        difference = abs(
            job_score - preferred_score
        )

        if difference == 0:
            return 1.0

        if difference == 1:
            return 0.5

        return 0.0

    # --------------------------------------------------
    # Location Score
    # --------------------------------------------------

    def location_score(
        self,
        job_location,
        preferred_location,
        is_remote=False
    ):

        if not preferred_location:
            return 0.5

        if not job_location:
            return 0.25

        preferred = preferred_location.lower().strip()
        job = job_location.lower().strip()

        if preferred in job:
            return 1.0

        if is_remote or "remote" in job:
            return 0.75

        location_parts = preferred.replace(
            ",",
            " "
        ).split()

        for part in location_parts:

            if len(part) >= 2 and part in job:
                return 0.65

        return 0.0

    # --------------------------------------------------
    # Education Score
    # --------------------------------------------------

    def education_score(
        self,
        job_title,
        job_description
    ):

        title = str(job_title).lower()
        description = str(job_description).lower()

        text = title + " " + description

        # PhD / doctoral positions
        if any(keyword in text for keyword in [
            "phd",
            "ph.d",
            "doctoral",
            "doctorate"
        ]):
            return 0.0

        # Graduate-specific positions
        if any(keyword in text for keyword in [
            "graduate intern",
            "graduate internship",
            "graduate student",
            "graduate students",
            "graduate masters",
            "graduate master's",
            "graduate master’s",
            "masters intern",
            "master's intern",
            "master’s intern",
            "masters internship",
            "master's internship",
            "master’s internship",
            "master's student",
            "masters student",
            "master’s student",
            "ms student",
            "m.s. student",
            "mba student"
        ]):
            return 0.1

        # Explicit undergraduate positions
        if any(keyword in text for keyword in [
            "undergraduate intern",
            "undergraduate internship",
            "undergraduate student",
            "undergraduate students",
            "bachelor's",
            "bachelors",
            "bachelor’s",
            "b.s.",
            "bs degree",
            "college student",
            "university student"
        ]):
            return 1.0

        # General internships
        if (
            "internship" in text
            or "intern" in title
        ):
            return 0.7

        return 0.5

    # --------------------------------------------------
    # Eligibility
    # --------------------------------------------------

    def is_eligible(
        self,
        job_title,
        job_description,
        preferred_experience
    ):

        if not preferred_experience:
            return True

        title = str(job_title).lower()
        description = str(job_description).lower()

        text = title + " " + description

        if preferred_experience == "Internship":

            graduate_only_keywords = [
                "phd",
                "ph.d",
                "doctoral",
                "doctorate",
                "graduate intern",
                "graduate internship",
                "graduate student",
                "graduate students",
                "graduate masters",
                "graduate master's",
                "graduate master’s",
                "masters intern",
                "master's intern",
                "master’s intern",
                "masters internship",
                "master's internship",
                "master’s internship",
                "master's student",
                "masters student",
                "master’s student",
                "ms student",
                "m.s. student",
                "mba student"
            ]

            if any(
                keyword in text
                for keyword in graduate_only_keywords
            ):
                return False

        return True

    # --------------------------------------------------
    # Role Relevance
    # --------------------------------------------------

    def role_relevance_score(
        self,
        job_title,
        job_description
    ):

        title = str(job_title).lower()
        description = str(job_description).lower()

        # Strong technical roles
        if any(keyword in title for keyword in [
            "machine learning",
            "ml engineer",
            "ai engineer",
            "artificial intelligence",
            "data scientist",
            "data science",
            "data engineer",
            "software engineer",
            "software engineering",
            "backend engineer",
            "backend developer",
            "full stack",
            "full-stack",
            "data analyst",
            "analytics engineer"
        ]):
            title_score = 1.0

        elif any(keyword in title for keyword in [
            "developer",
            "programmer",
            "software development",
            "technical",
            "technology"
        ]):
            title_score = 0.75

        elif any(keyword in title for keyword in [
            "analyst",
            "analytics"
        ]):
            title_score = 0.65

        else:
            title_score = 0.2

        technical_keywords = [
            "machine learning",
            "artificial intelligence",
            "data science",
            "data engineering",
            "software engineering",
            "python",
            "java",
            "sql",
            "scikit-learn",
            "pytorch",
            "tensorflow",
            "pandas",
            "numpy",
            "api",
            "rest api",
            "backend",
            "deep learning",
            "nlp",
            "computer vision",
            "statistics",
            "regression"
        ]

        technical_matches = sum(
            1
            for keyword in technical_keywords
            if keyword in description
        )

        if technical_matches >= 6:
            description_score = 1.0

        elif technical_matches >= 4:
            description_score = 0.85

        elif technical_matches >= 2:
            description_score = 0.7

        elif technical_matches >= 1:
            description_score = 0.5

        else:
            description_score = 0.2

        return (
            title_score * 0.75
            + description_score * 0.25
        )

    # --------------------------------------------------
    # Generate Match Explanation
    # --------------------------------------------------

    def generate_match_explanation(
        self,
        resume_text,
        job_title,
        job_description,
        job_level,
        preferred_level,
        job_location,
        preferred_location,
        is_remote=False
    ):
        """
        Generate a short, deterministic explanation describing
        why a job matches the resume.

        This intentionally uses keyword overlap rather than an
        LLM so the explanation is grounded in the actual resume
        and job description.
        """

        resume_lower = str(
            resume_text
        ).lower()

        job_text = (
            str(job_title)
            + " "
            + str(job_description)
        ).lower()

        # Important technical skills
        skill_groups = {
            "Python": ["python"],
            "Java": ["java"],
            "SQL": ["sql"],
            "R": ["\nr ", " r,", "r programming", "r language"],
            "Spring Boot": ["spring boot"],
            "REST APIs": [
                "rest api",
                "rest apis"
            ],
            "Machine Learning": [
                "machine learning",
                "ml"
            ],
            "Artificial Intelligence": [
                "artificial intelligence",
                "ai"
            ],
            "Data Science": [
                "data science",
                "data scientist"
            ],
            "Data Analysis": [
                "data analysis",
                "data analytics",
                "data analyst"
            ],
            "Pandas": ["pandas"],
            "NumPy": ["numpy"],
            "PyTorch": ["pytorch"],
            "TensorFlow": ["tensorflow"],
            "Scikit-learn": [
                "scikit-learn",
                "sklearn"
            ],
            "APIs": ["api", "apis"],
            "Backend Development": [
                "backend",
                "back-end"
            ],
            "Git": ["git", "github"],
            "Statistics": [
                "statistics",
                "statistical"
            ],
            "Regression": ["regression"],
            "Cloud": [
                "aws",
                "azure",
                "gcp",
                "cloud"
            ]
        }

        matched_skills = []

        for skill, keywords in skill_groups.items():

            resume_has_skill = any(
                keyword in resume_lower
                for keyword in keywords
            )

            job_has_skill = any(
                keyword in job_text
                for keyword in keywords
            )

            if resume_has_skill and job_has_skill:
                matched_skills.append(skill)

        # Remove overly generic AI/API matches if better
        # specific skills exist
        if len(matched_skills) > 5:
            matched_skills = matched_skills[:5]

        reasons = []

        # Role match
        title_lower = str(
            job_title
        ).lower()

        role_keywords = [
            "software engineer",
            "software engineering",
            "data scientist",
            "data science",
            "data analyst",
            "data analytics",
            "machine learning",
            "ai engineer",
            "backend",
            "developer",
            "analytics"
        ]

        if any(
            keyword in title_lower
            for keyword in role_keywords
        ):
            reasons.append(
                f"Role aligns with your technical background"
            )

        # Skill overlap
        if matched_skills:

            if len(matched_skills) == 1:

                reasons.append(
                    f"Matches your {matched_skills[0]} experience"
                )

            else:

                skills_text = ", ".join(
                    matched_skills
                )

                reasons.append(
                    f"Matches your skills in {skills_text}"
                )

        # Experience
        if preferred_level:

            if job_level == preferred_level:

                reasons.append(
                    f"Matches your selected {preferred_level} experience level"
                )

            elif job_level == "Unknown":

                reasons.append(
                    "Experience level was not specified in the listing"
                )

        # Location
        if preferred_location:

            if (
                preferred_location.lower()
                in str(job_location).lower()
            ):

                reasons.append(
                    f"Located in your preferred area"
                )

            elif is_remote:

                reasons.append(
                    "Offers a remote work option"
                )

        # Fallback
        if not reasons:

            reasons.append(
                "Strong semantic match with your resume"
            )

        # Keep explanation concise
        reasons = reasons[:4]

        return reasons

    # --------------------------------------------------
    # Search Jobs
    # --------------------------------------------------

    def search_jobs(
        self,
        resume_text,
        jobs=None,
        k=10,
        preferred_location=None,
        preferred_experience=None
    ):

        if jobs is None or len(jobs) == 0:
            return pd.DataFrame()

        search_jobs_df = jobs.copy()

        # Make sure expected columns exist
        for column in [
            "job_id",
            "title",
            "company",
            "description",
            "location",
            "experience_level",
            "is_remote",
            "url"
        ]:

            if column not in search_jobs_df.columns:
                search_jobs_df[column] = ""

        # --------------------------------------------------
        # 1. Eligibility Filtering
        # --------------------------------------------------

        if preferred_experience:

            eligible_mask = search_jobs_df.apply(
                lambda row: self.is_eligible(
                    row.get("title", ""),
                    row.get("description", ""),
                    preferred_experience
                ),
                axis=1
            )

            search_jobs_df = search_jobs_df[
                eligible_mask
            ].copy()

        if len(search_jobs_df) == 0:
            return pd.DataFrame()

        # --------------------------------------------------
        # 2. Create Job Text
        # --------------------------------------------------

        combined_text = (
            search_jobs_df["title"].fillna("")
            + " "
            + search_jobs_df["company"].fillna("")
            + " "
            + search_jobs_df["description"].fillna("")
        ).tolist()

        # --------------------------------------------------
        # 3. Resume Embedding
        # --------------------------------------------------

        resume_embedding = self.encode_text(
            [resume_text]
        )

        # --------------------------------------------------
        # 4. Job Embeddings
        # --------------------------------------------------

        job_embeddings = self.encode_text(
            combined_text
        )

        # --------------------------------------------------
        # 5. FAISS Similarity Search
        # --------------------------------------------------

        index = faiss.IndexFlatIP(
            job_embeddings.shape[1]
        )

        index.add(job_embeddings)

        candidate_count = min(
            max(k * 10, 50),
            len(search_jobs_df)
        )

        similarities, indices = index.search(
            resume_embedding,
            candidate_count
        )

        valid_indices = [
            i
            for i in indices[0]
            if i >= 0
        ]

        matched_jobs = search_jobs_df.iloc[
            valid_indices
        ].copy()

        matched_jobs["semantic_similarity"] = similarities[0][
            :len(matched_jobs)
        ]

        # --------------------------------------------------
        # 6. Semantic Score
        # --------------------------------------------------

        matched_jobs["semantic_score"] = (
            matched_jobs["semantic_similarity"] + 1
        ) / 2

        # --------------------------------------------------
        # 7. Location Score
        # --------------------------------------------------

        if preferred_location:

            matched_jobs["location_match"] = matched_jobs.apply(
                lambda row: self.location_score(
                    row.get("location", ""),
                    preferred_location,
                    row.get("is_remote", False)
                ),
                axis=1
            )

        else:

            matched_jobs["location_match"] = 0.5

        # --------------------------------------------------
        # 8. Experience Score
        # --------------------------------------------------

        if preferred_experience:

            matched_jobs["experience_match"] = matched_jobs.apply(
                lambda row: self.experience_score(
                    row.get(
                        "experience_level",
                        "Unknown"
                    ),
                    preferred_experience
                ),
                axis=1
            )

        else:

            matched_jobs["experience_match"] = 0.5

        # --------------------------------------------------
        # 9. Education Score
        # --------------------------------------------------

        matched_jobs["education_match"] = matched_jobs.apply(
            lambda row: self.education_score(
                row.get("title", ""),
                row.get("description", "")
            ),
            axis=1
        )

        # --------------------------------------------------
        # 10. Role Relevance Score
        # --------------------------------------------------

        matched_jobs["role_match"] = matched_jobs.apply(
            lambda row: self.role_relevance_score(
                row.get("title", ""),
                row.get("description", "")
            ),
            axis=1
        )

        # --------------------------------------------------
        # 11. Final Recommendation Score
        # --------------------------------------------------

        matched_jobs["final_score"] = (
            matched_jobs["semantic_score"] * 0.55
            + matched_jobs["role_match"] * 0.20
            + matched_jobs["location_match"] * 0.10
            + matched_jobs["experience_match"] * 0.05
            + matched_jobs["education_match"] * 0.10
        )

        # --------------------------------------------------
        # 12. User-Facing Match Score
        # --------------------------------------------------

        matched_jobs["similarity"] = (
            matched_jobs["final_score"] * 100
        ).round(1)

        # --------------------------------------------------
        # 13. Sort
        # --------------------------------------------------

        matched_jobs = matched_jobs.sort_values(
            by="final_score",
            ascending=False
        )

        # --------------------------------------------------
        # 14. Deduplicate
        # --------------------------------------------------

        if "job_id" in matched_jobs.columns:

            matched_jobs = matched_jobs.drop_duplicates(
                subset=["job_id"],
                keep="first"
            )

        matched_jobs = matched_jobs.drop_duplicates(
            subset=["title", "company"],
            keep="first"
        )

        # --------------------------------------------------
        # 15. Generate Match Explanations
        # --------------------------------------------------

        matched_jobs["match_reasons"] = matched_jobs.apply(
            lambda row: self.generate_match_explanation(
                resume_text=resume_text,
                job_title=row.get("title", ""),
                job_description=row.get("description", ""),
                job_level=row.get(
                    "experience_level",
                    "Unknown"
                ),
                preferred_level=preferred_experience,
                job_location=row.get(
                    "location",
                    ""
                ),
                preferred_location=preferred_location,
                is_remote=row.get(
                    "is_remote",
                    False
                )
            ),
            axis=1
        )

        # --------------------------------------------------
        # 16. Return Results
        # --------------------------------------------------

        expected_columns = [
            "job_id",
            "title",
            "company",
            "location",
            "experience_level",
            "similarity",
            "match_reasons",
            "semantic_similarity",
            "role_match",
            "location_match",
            "experience_match",
            "education_match",
            "final_score",
            "url"
        ]

        existing_columns = [
            column
            for column in expected_columns
            if column in matched_jobs.columns
        ]

        return matched_jobs.head(k)[
            existing_columns
        ]