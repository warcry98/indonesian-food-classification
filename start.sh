#!/bin/bash
set -e

uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

streamlit run app.py &
STREAMLIT_PID=$!

trap "kill $UVICORN_PID $STREAMLIT_PID" SIGTERM SIGINT

wait $UVICORN_PID $STREAMLIT_PID