from fastapi import FastAPI
from .main import app 

# This is the line Vercel needs
handler = app
