
# Azure Agentic AI

A hands-on Python project for learning Agentic AI
development, API design, testing, and Azure deployment.

## Current features

- FastAPI application
- Root endpoint
- Health-check endpoint
- Automated API tests

## Setup

Create and activate a Python virtual environment,
then install dependencies:

    python -m pip install -r requirements.txt

## Run

    python -m uvicorn app.main:app --reload

## Test

    python -m pytest -v

## API documentation

Open http://127.0.0.1:8000/docs
while the application is running.
