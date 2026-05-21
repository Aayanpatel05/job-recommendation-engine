AI Job Recommendation Engine

An AI-powered job recommendation platform that analyzes a user’s resume, dynamically fetches live jobs, and recommends the most relevant positions using semantic similarity with transformer embeddings and FAISS vector search.

Features
Resume upload support (PDF, DOCX, TXT)
Automatic resume text extraction
AI-powered semantic job matching
Live job fetching using the Adzuna API
Dynamic query generation from resumes
FAISS vector similarity search
Location and experience-level score boosting
React frontend + FastAPI backend
Real-time recommendations
Tech Stack
Backend
Python
FastAPI
Sentence Transformers
FAISS
Pandas
spaCy
PDFMiner
Frontend
React
JavaScript
Fetch API
APIs
Adzuna Jobs API
How It Works
User uploads a resume
Resume text is extracted and cleaned
NLP generates intelligent search queries from the resume
Live jobs are fetched from the Adzuna API
Job descriptions and resume are converted into embeddings using all-MiniLM-L6-v2
FAISS performs semantic similarity search
Results are boosted based on:
Preferred location
Experience level
Top matching jobs are returned to the frontend
Project Structure
job-recommendation-engine/
│
├── backend/
│   ├── src/
│   │   ├── api/
│   │   ├── core/
│   │   ├── pipeline/
│   │   └── utils/
│   │
│   ├── data/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   └── package.json
│
└── README.md
Installation
1. Clone Repository
git clone <your-repo-url>
cd job-recommendation-engine
Backend Setup
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Backend Dependencies
pip install -r requirements.txt
4. Download spaCy Model
python -m spacy download en_core_web_sm
5. Configure Adzuna API Keys

Inside job_fetcher.py:

APP_ID = "YOUR_APP_ID"
APP_KEY = "YOUR_APP_KEY"

You can get free API keys from:

Adzuna Developer Portal

6. Run FastAPI Backend
uvicorn src.api.main:app --reload

Backend will run at:

http://127.0.0.1:8000

Swagger docs:

http://127.0.0.1:8000/docs
Frontend Setup
7. Install Frontend Dependencies
cd frontend
npm install
8. Run React Frontend
npm run dev

Frontend usually runs at:

http://localhost:5173
API Endpoints
Health Check
GET /

Response:

{
  "message": "Job Recommendation API running"
}
Recommend Jobs
POST /recommend
Form Data
Field	Type	Required
file	UploadFile	Yes
Query Parameters
Parameter	Description
top_k	Number of recommendations
preferred_location	Preferred job location
experience_level	Desired experience level
Example Request
curl -X POST \
"http://127.0.0.1:8000/recommend?top_k=10&preferred_location=Atlanta&experience_level=Entry%20level" \
-F "file=@resume.pdf"
Example Response
{
  "source": "dynamic_live_jobs",
  "total_jobs_fetched": 120,
  "recommendations": [
    {
      "title": "Machine Learning Engineer",
      "location": "Atlanta, Georgia",
      "similarity": 0.82
    }
  ]
}
AI / NLP Features
Resume Query Generation

The system extracts noun phrases from resumes using spaCy and converts them into intelligent job search queries.

Example generated queries:

machine learning
data science
python developer
artificial intelligence
statistical analysis
Semantic Similarity Search

The project uses:

sentence-transformers/all-MiniLM-L6-v2
FAISS vector indexing

This allows matching based on semantic meaning instead of keyword matching.

Performance Optimizations

Implemented optimizations include:

Global model loading
Reusable FAISS engine
Dynamic live job retrieval
Reduced duplicate embedding generation
Query filtering using stop words
Score boosting instead of hard filtering
Future Improvements
Authentication system
Save favorite jobs
User profiles
Resume feedback scoring
Fine-tuned recommendation models
GPU acceleration
Docker deployment
Cloud deployment (AWS/GCP/Azure)
Redis caching
Background task queues
Screenshots

Add screenshots of:

Resume upload page
Recommended jobs page
Swagger API docs
