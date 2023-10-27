from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
import json
from panthera import (
    save_user_session, 
    get_user_session, 
    log_message,
    reset_chatm,
    llm_request
    )

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
    message = await request.json()
    logger.info(message)
    """
    INFO:server:{
       'message_id': 22,
       'from': {
            'id': 106129214, 
            'is_bot': False, 
            'first_name': 'Alex', 
            'username': 'format37', 
            'language_code': 'en', 
            'is_premium': True
         }, 
        'chat': {
            'id': 106129214, 
            'first_name': 'Alex', 
            'username': 'format37', 
            'type': 'private'
        }, 
        'date': 1698311200, 
        'text': '9'
    }
    """
    log_message(message)
    chat_id = message['chat']['id']
    text = message['text']
    user_session = get_user_session(message['from']['id'])
    logger.info(f'user_session: {user_session}')
    answer = 'empty'
    # if message text is /reset
    if message['text'] == '/reset':
        reset_chat(message['chat']['id'])
        answer = 'Chat messages has been forgotten'

    # if message text is /start
    elif message['text'] == '/start':
        answer = 'Welcome to the bot'

    else
        answer = llm_request(user_session, chat_id, text)

    return JSONResponse(content={
        "status": "ok",
        "message": answer
        })
