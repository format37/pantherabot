from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import telebot
import os
import logging
# import ssl

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
