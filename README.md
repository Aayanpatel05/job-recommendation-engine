# AI Job Recommendation Engine

An AI-powered job recommendation system that matches resumes to relevant job listings using NLP, sentence embeddings, and vector similarity search.

The application extracts text from a user’s resume, generates intelligent search queries, fetches live jobs from the Adzuna API, and ranks the jobs using semantic similarity with FAISS and Sentence Transformers.

---

# Features

- Upload resumes in PDF, DOCX, or TXT format
- Extract and clean resume text automatically
- Generate dynamic job search queries from resume content
- Fetch live jobs from the Adzuna Jobs API
- Rank jobs using semantic similarity
- Boost recommendations using:
  - Preferred location
  - Experience level
- Fast vector search with FAISS
- React frontend + FastAPI backend

---

# Tech Stack

## Frontend
- React
- JavaScript
- Fetch API

## Backend
- FastAPI
- Python

## Machine Learning / NLP
- Sentence Transformers
- FAISS
- spaCy

## APIs
- Adzuna Jobs API

---

# Project Structure

```bash
job-recommendation-engine/
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
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
└── README.md
```

---

# How It Works

## 1. Resume Upload
The user uploads a resume through the React frontend.

## 2. Resume Parsing
The backend extracts and cleans the resume text.

## 3. Query Generation
spaCy analyzes the resume and generates multiple search queries based on important noun phrases.

Example:
```python
[
  "machine learning",
  "data science",
  "python developer",
  "analytics projects"
]
```

## 4. Live Job Fetching
The system fetches relevant jobs from the Adzuna API using the generated queries.

## 5. Semantic Matching
The resume and job descriptions are converted into embeddings using:

```python
all-MiniLM-L6-v2
```

## 6. FAISS Similarity Search
FAISS ranks the jobs based on vector similarity.

## 7. Score Boosting
Additional boosts are applied for:
- Matching locations
- Matching experience levels

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

Create a `.env` file:

```env
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

---

# Running the Application

## Start Backend

From project root:

```bash
uvicorn src.api.main:app --reload
```

Backend runs on:

```bash
http://127.0.0.1:8000
```

---

## Start Frontend

```bash
cd frontend

npm run dev
```

Frontend runs on:

```bash
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

| Field | Type |
|---|---|
| file | Resume file |

### Query Parameters

| Parameter | Description |
|---|---|
| top_k | Number of recommendations |
| preferred_location | Optional location boost |
| experience_level | Optional experience boost |

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
      "title": "Data Science Intern",
      "location": "Atlanta, Georgia",
      "similarity": 0.82
    }
  ]
}
```

---

# Machine Learning Details

## Embedding Model

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Used to convert:
- resumes
- job descriptions

into semantic vector embeddings.

---

## Similarity Search

FAISS is used for:
- fast nearest-neighbor search
- scalable semantic retrieval

---

# Future Improvements

- User authentication
- Save favorite jobs
- Better resume parsing
- GPU acceleration
- Caching embeddings
- Better query ranking
- Support multiple job APIs
- Fine-tuned transformer models
- Deployment with Docker + AWS

---

# Author

Aayan Patel

Data Science Student

---

# License

MIT License
