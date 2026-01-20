# PlacementSprint (Pydantic AI + FastAPI + OpenRouter)

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
uvicorn src.index:app --reload --port 8000
