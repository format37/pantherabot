from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
import json
"""from panthera import (
    save_user_session, 
    get_user_session, 
    log_message,
    reset_chat,
    llm_request
    )"""
from panthera import Panthera

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

    panthera = Panthera()

    panthera.log_message(message)
    chat_id = message['chat']['id']
    text = message['text']

    user_session = panthera.get_user_session(message['from']['id'])
    logger.info(f'user_session: {user_session}')
    answer = 'empty'
    # if message text is /reset
    if message['text'] == '/reset':
        panthera.reset_chat(message['chat']['id'])
        answer = 'Chat messages has been forgotten'

    # if message text is /start
    elif message['text'] == '/start':
        answer = 'Welcome to the bot'

    else:
        response = panthera.llm_request(user_session, chat_id, text)
        if response.status_code == 200:
            # response.text
            """
            {
                "id":"chatcmpl-8EEabpufU95pSk2tOg29tDP2zOXgN",
                "object":"chat.completion",
                "created":1698403365,
                "model":"gpt-4-0613",
                "choices":[
                    {
                        "index":0,"message":
                        {
                            "role":"assistant",
                            "content":"Red is one of the primary colors, along with blue and yellow. It's the color of blood, rubies, and strawberries. Next to orange at the end of the visible light spectrum, it's typically associated with energy, danger, strength, power, determination, as well as passion, desire, and love. It has a wavelength of approximately 625–740 nanometers on the electromagnetic spectrum."
                        },
                        "finish_reason":"stop"
                    }
                ],
                "usage":
                {
                    "prompt_tokens":21,
                    "completion_tokens":81,
                    "total_tokens":102
                }
            }
            """
            response_json = json.loads(response.text)
            logger.info(f'response_json: {response_json}')
            answer = response_json['choices'][0]['message']['content']
        else:
            logger.error(f'{response.status_code}: LLM request unsuccessfull: {response.text}')
            answer = 'unable to read response'

    return JSONResponse(content={
        "status": "ok",
        "message": str(answer)
        })
