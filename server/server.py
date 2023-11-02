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


def get_keyboard(user_session, current_screen):


    with open('data/menu.json') as f:
        menu = json.load(f)

    # current_screen = user_session['last_cmd']

    if current_screen in menu:
        # buttons = menu[current_screen]['buttons']
        message = menu[current_screen]['message']

        # Format message with current values if needed
        if '%s' in message:  
            if current_screen == 'Model':
                model = user_session['model']
            elif current_screen == 'Language':
                lang = user_session['language']
            message = message % model if 'model' in locals() else message
            message = message % lang if 'language' in locals() else message
            menu[current_screen]['message'] = message

        # return {'message': message, 'buttons': buttons}
        return menu[current_screen]

    else:
        # Default to start screen
        return menu['Default']


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

    # elif message['text'] == '/configure': # TODO: account the non-private chats
    elif user_session['last_cmd'] != 'start':
        """keyboard_dict = {
            'message': 'Configuration',
            'row_width': 1,
            'resize_keyboard': True,
            'buttons': [
                    {
                    'text': "Model",
                    'request_contact': False
                    }
                ]
        }"""
        keyboard_dict = get_keyboard(user_session, message['text'])

        logger.info(f'keyboard_dict: {keyboard_dict}')

        # Get user session
        # user_session = panthera.get_user_session(message['from']['id'])
        # Update user session
        user_session['last_cmd'] = message['text']
        # Save user session
        panthera.save_user_session(message['from']['id'], user_session)

        return JSONResponse(content={
            "type": "keyboard",
            "body": keyboard_dict
            })

    else:
        answer = panthera.llm_request(user_session, message)

    return JSONResponse(content={
        "type": "text",
        "body": str(answer)
        })
