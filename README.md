# DocuMind: Enterprise RAG with Uncertainty Quantification & Corrective Retrieval

DocuMind is an enterprise-grade Retrieval-Augmented Generation (RAG) platform equipped with advanced uncertainty quantification:
- **Semantic Entropy (SE)**: Replaces verbalized confidence by clustering stochastically sampled answers and computing entropy over semantic meaning equivalence classes (separating aleatoric and epistemic uncertainty).
- **Evidential Deep Learning (EDL)**: Replaces heuristic linear weighting with a Dirichlet belief-mass fusion engine ($u + \sum b_k = 1$), computing predictive confidence, belief masses, and vacuity (epistemic uncertainty).
- **Split Conformal Prediction**: Replaces brittle static thresholds with statistically guaranteed risk control ($1 - \alpha$ target coverage) and dynamic abstention thresholds.
- **Corrective Retrieval & Reranking**: Query expansion, dense semantic retrieval via ChromaDB, and Cohere neural reranking.

---

## Prerequisites

- **Python**: 3.10+ (Existing virtualenv is at `backend/venv`)
- **Node.js**: 18+ and `npm`
- **API Keys**:
  - `GROQ_API_KEY` (LLM inference)
  - `COHERE_API_KEY` (Reranking)

---

## Quick Start: How to Run the Project

You need two terminal windows: one for the **Backend** and one for the **Frontend**.

### Terminal 1: Backend (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Activate the Python virtual environment:

   - **If you use Fish shell:**
     ```fish
     source venv/bin/activate.fish
     ```
   - **If you use Bash or Zsh:**
     ```bash
     source venv/bin/activate
     ```

3. Ensure dependencies and environment variables are set:
   - Make sure `.env` contains your keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     COHERE_API_KEY=your_cohere_api_key
     ```
   - (Optional) Install dependencies if setting up a new environment:
     ```bash
     pip install -r requirements.txt
     ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   *(Alternatively: `python main.py`)*

   - **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
   - **Conformal Calibration Status**: [http://localhost:8000/conformal/status](http://localhost:8000/conformal/status)

---

### Terminal 2: Frontend (React + Vite)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies (first time only):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser:
   - **Frontend UI**: [http://localhost:5173](http://localhost:5173)

---

## Running Evaluation & Benchmarks

To run the calibration and evidential fusion benchmark (evaluating ECE, Brier score, NLL, AUROC, AUPRC, and selective risk):

```bash
cd backend
# With venv activated:
python evaluate_edl.py
```

This compares Dirichlet Evidential Deep Learning (EDL) fusion directly against the baseline linear fusion model across 200 calibration points.

---

## Core API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload and ingest PDF/text documents into ChromaDB |
| `GET` | `/documents/list` | List all ingested documents |
| `POST` | `/query/uncertainty` | Query the RAG engine with Semantic Entropy, EDL, and Conformal thresholds |
| `POST` | `/query/feedback` | Submit ground-truth feedback to calibrate conformal predictor |
| `GET` | `/conformal/status` | Check calibration size, quantile threshold, and coverage status |
| `GET` | `/health` | Check backend service and DB status |

---

## Project Structure

```
major/
├── README.md                      # Project run guide and overview
├── backend/
│   ├── .env                       # API keys and hyperparameters
│   ├── main.py                    # FastAPI entrypoint (port 8000)
│   ├── Confidence.py              # Semantic Entropy & Confidence estimation
│   ├── evidential.py              # Dirichlet Evidential Deep Learning fusion engine
│   ├── conformal.py               # Split conformal prediction and calibration store
│   ├── corrective.py              # Query expansion and reranking
│   ├── evaluate_edl.py            # Calibration evaluation benchmark
│   ├── requirements.txt           # Python dependencies
│   ├── app/
│   │   ├── api/router.py          # API routes and conformal endpoints
│   │   ├── core/config.py         # App settings & conformal hyperparams
│   │   ├── core/database.py       # ChromaDB & ConformalPredictor instances
│   │   ├── models/schemas.py      # Pydantic schemas (SE, EDL, Conformal)
│   │   └── services/rag_service.py# RAG pipeline orchestration
│   └── venv/                      # Python virtual environment
└── frontend/
    ├── package.json               # Node.js dependencies
    ├── vite.config.ts             # Vite configuration
    └── src/
        ├── App.tsx                # Main UI with SE, EDL & Conformal cards
        ├── App.css                # Custom dark mode UI stylesheet
        └── index.css              # Global tokens and base styling
```
