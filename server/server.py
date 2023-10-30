from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
import json
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
        answer = 'Chat messages memory has been cleaned'

    # if message text is /start
    elif message['text'] == '/start':
        answer = 'Welcome to the bot'

    elif message['text'] == '/configure': # TODO: account the non-private chats
        keyboard_dict = {
            'message': 'Configuration',
            'row_width': 1,
            'resize_keyboard': True,
            'buttons': [
                    {
                    'text': "Model",
                    'request_contact': False
                    }
                ]
        }

        return JSONResponse(content={
            "type": "keyboard",
            "body": keyboard_dict
            })

    else:
        answer = panthera.llm_request(user_session, message)
        """if response.status_code == 200:            
            response_json = json.loads(response.text)
            logger.info(f'response_json: {response_json}')
            answer = response_json['choices'][0]['message']['content']
        else:
            logger.error(f'{response.status_code}: LLM request unsuccessfull: {response.text}')
            answer = 'unable to read response'"""

    return JSONResponse(content={
        "type": "text",
        "body": str(answer)
        })
