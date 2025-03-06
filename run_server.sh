#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
uvicorn api:app --reload 