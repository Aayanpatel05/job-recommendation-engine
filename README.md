# AI Job Recommendation Engine

An AI-powered job recommendation system that matches resumes to relevant **technology job opportunities** using NLP, sentence embeddings, and vector similarity search.

The application extracts information from a user's resume, generates multiple role-specific search queries, retrieves live job listings from the Adzuna API, and ranks opportunities using semantic similarity and personalized factors such as role relevance, location, experience level, and education.

---

# Features

* Upload resumes in **PDF, DOCX, or TXT** format
* Automatically extract and clean resume text
* Generate up to **5 role-specific job search queries**
* Fetch live job listings from the **Adzuna Jobs API**
* Retrieve up to **1,250 job listings** across generated queries before deduplication
* Remove duplicate job postings
* Generate semantic embeddings using **Sentence Transformers**
* Perform fast vector similarity search using **FAISS**
* Rank jobs using multiple personalized factors:

  * Semantic similarity
  * Role relevance
  * Preferred location
  * Experience level
  * Education requirements
* Filter jobs that do not match the requested experience level
* Detect internship, entry-level, associate, mid-senior, director, and executive roles
* Identify remote job opportunities
* Generate explainable **"Why this job matches"** recommendations
* Provide direct application links for recommended jobs
* Return the **top 10 personalized recommendations**
* React frontend + FastAPI backend

---

# Tech Stack

## Frontend

* React
* JavaScript
* Fetch API

## Backend

* FastAPI
* Python
* Pandas

## Machine Learning / NLP

* Sentence Transformers
* `all-MiniLM-L6-v2`
* FAISS
* spaCy
* NumPy

## APIs

* Adzuna Jobs API

---

# Project Structure

```bash
job-recommendation-engine/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   └── package.json
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── core/
│   │   ├── recommendation_engine.py
│   │   ├── resume_parser.py
│   │   ├── query_generator.py
│   │   └── job_fetcher.py
│   │
│   └── pipeline/
│       └── scheduler.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# How It Works

## 1. Resume Upload

The user uploads a resume through the React frontend.

Supported formats:

* PDF
* DOCX
* TXT

The frontend sends the resume to the FastAPI `/recommend` endpoint.

---

## 2. Resume Parsing

The backend extracts text from the uploaded resume and cleans it before processing.

The cleaned resume is used for both query generation and semantic matching.

---

## 3. Dynamic Query Generation

The system analyzes the resume and generates up to **5 role-specific search queries** based on relevant skills, roles, and technical experience.

Example:

```python
[
    "machine learning",
    "data science",
    "python developer",
    "data analyst",
    "software engineer"
]
```

The queries are designed to retrieve different but relevant technology roles rather than relying on a single search query.

---

## 4. Live Job Retrieval

Each generated query is sent to the **Adzuna Jobs API**.

The system can retrieve up to:

```text
5 queries × 5 pages × 50 jobs = 1,250 jobs
```

The results are then deduplicated using the Adzuna job ID.

The system also extracts:

* Job title
* Company
* Location
* Description
* Experience level
* Remote status
* Application URL

---

## 5. Experience Detection

Job experience levels are primarily detected from the job title.

The system recognizes categories including:

```text
Internship
Entry level
Associate
Mid-Senior level
Director
Executive
Unknown
```

Unknown experience levels are preserved rather than automatically discarded, allowing the recommendation engine to evaluate them during ranking.

---

## 6. Semantic Matching

The resume and job descriptions are converted into vector embeddings using:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

This allows the system to compare the **meaning and context** of the resume against job descriptions rather than relying only on exact keyword matches.

---

## 7. FAISS Similarity Search

FAISS is used to efficiently search the job embedding vectors for opportunities that are semantically similar to the resume.

The system retrieves a candidate pool of relevant jobs before applying additional ranking criteria.

---

## 8. Personalized Ranking

The recommendation engine combines multiple signals when ranking jobs:

```text
Semantic Similarity      55%
Role Relevance            20%
Location Match            10%
Education Match           10%
Experience Match           5%
```

This allows the final ranking to consider more than just similarity between the resume and job description.

---

## 9. Eligibility Filtering

Before final ranking, the system checks whether jobs meet the requested experience and education requirements.

For example, when searching for internships, graduate-only internship positions can be excluded while jobs with unclear experience requirements remain eligible for evaluation.

---

## 10. Explainable Recommendations

Each recommendation includes personalized reasons explaining why the job was selected.

Example:

```text
Why this job matches:

• Matches Python experience
• Relevant machine learning responsibilities
• Technology-focused role
• Matches preferred location
```

This makes the recommendation results easier for users to understand instead of presenting only a numerical match score.

---

## 11. Application Links

Each job recommendation includes the original application URL when available.

Users can click:

```text
Apply for this job →
```

to open the job listing in a new browser tab.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/job-recommendation-engine.git

cd job-recommendation-engine
```

---

# Backend Setup

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# Frontend Setup

```bash
cd frontend

npm install
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

Do not commit your `.env` file or API credentials to GitHub.

---

# Running the Application

## Start Backend

From the project root:

```bash
uvicorn src.api.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

## Start Frontend

In a separate terminal:

```bash
cd frontend

npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

# API Endpoints

## Health Check

```http
GET /
```

Response:

```json
{
  "message": "Job Recommendation API running"
}
```

---

## Recommend Jobs

```http
POST /recommend
```

### Form Data

| Field  | Type | Description                        |
| ------ | ---- | ---------------------------------- |
| `file` | File | Resume in PDF, DOCX, or TXT format |

### Query Parameters

| Parameter            | Description                         |
| -------------------- | ----------------------------------- |
| `top_k`              | Number of recommendations to return |
| `preferred_location` | Optional preferred location         |
| `experience_level`   | Optional desired experience level   |

---

# Example Request

```bash
curl -X POST \
  "http://127.0.0.1:8000/recommend?top_k=10&preferred_location=Atlanta&experience_level=Entry%20level" \
  -F "file=@resume.pdf"
```

---

# Example Response

```json
{
  "source": "dynamic_live_jobs",
  "total_jobs_fetched": 82,
  "recommendations": [
    {
      "job_id": "123456",
      "title": "Data Science Intern",
      "company": "Example Company",
      "location": "Atlanta, Georgia",
      "experience_level": "Internship",
      "similarity": 82.4,
      "url": "https://example.com/job",
      "match_reasons": [
        "Matches Python experience",
        "Relevant data science responsibilities",
        "Matches preferred location"
      ]
    }
  ]
}
```

---

# Machine Learning Details

## Embedding Model

The system uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

The model converts resumes and job descriptions into dense vector representations that capture semantic relationships between text.

---

## Vector Search

FAISS is used for efficient nearest-neighbor search over job embeddings.

The general pipeline is:

```text
Resume
   ↓
Resume Embedding
   ↓
FAISS Similarity Search
   ↓
Candidate Jobs
   ↓
Personalized Ranking
   ↓
Top Recommendations
```

---

# Evaluation

The recommendation engine was manually evaluated using technical and data science resume test cases.

Example Precision@10 results:

| Test Case                       | Precision@10 |
| ------------------------------- | -----------: |
| Associate Data Analytics        |          90% |
| Mid-Senior Software Engineering |         100% |

These results represent individual test cases and are not intended to represent overall production accuracy.

---

# Future Improvements

* User authentication
* Save and favorite jobs
* Improved resume parsing
* Embedding caching
* Query quality optimization
* Additional job APIs
* Fine-tuned transformer models
* Improved ranking evaluation
* Automated recommendation evaluation
* Docker deployment
* AWS deployment
* Persistent user profiles and preferences

---

# Author

**Aayan Patel**

Data Science Student

---

# License

MIT License
