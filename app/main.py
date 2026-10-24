from fastapi import FastAPI

app = FastAPI(
    title="Azure Agentic AI",
    description="My hands-on AI engineering learning project",
    version="0.1.0",
)


@app.get("/courses")
def root():
    return {
        "message": "Welcome my Azure Agentic AI project!",
        "lesson": 1,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "azure-agentic-ai",
    }