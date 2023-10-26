from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import logging

# Initialize FastAPI
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@app.get("/test")
async def call_test():
    logger.info('call_test')
    return JSONResponse(content={"status": "ok"})

@app.post("/message")
async def call_message(request: Request):
    logger.info('call_message')
    body = await request.json()
    logger.info(body)
    return JSONResponse(content={"status": "ok"})