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
            elif current_screen == 'Topic':
                topic = user_session['topic']
            message = message % model if 'model' in locals() else message
            message = message % lang if 'language' in locals() else message
            message = message % topic if 'topic' in locals() else message
            menu[current_screen]['message'] = message

        # return {'message': message, 'buttons': buttons}
        return menu[current_screen]

    else:
        # Default to start screen
        return menu['Default']
    

def get_message_type(user_session, text):
    if text == '/start':
        return 'cmd'
    elif text == '/configure':
        return 'cmd'
    elif text == '/reset':
        return 'cmd'
    # if user_session['last_cmd'] != 'text':
    # Check the buttons
    with open('data/menu.json') as f:
        menu = json.load(f)
    for key, value in menu.items():
        # logger.info(f'key: {key}, value: {value}')
        if text == key:
            return 'button'
        for button in value['buttons']:
            # logger.info(f'button: {button}')
            if text == button['text']:
                return 'button'
    return 'text'


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
    message_type = get_message_type(user_session, text)
    logger.info(f'message_type: {message_type}')
    # if message text is /reset
    if message['text'] == '/reset':
        panthera.reset_chat(message['chat']['id'])
        answer = 'Chat messages memory has been cleaned'

    # if message text is /start
    elif message['text'] == '/start':
        answer = 'Welcome to the bot'

    # elif message['text'] == '/configure': # TODO: account the non-private chats
    # elif user_session['last_cmd'] != 'start':
    
    elif message_type == 'button':
        
        keyboard_dict = get_keyboard(user_session, message['text'])

        # Model
        if user_session['last_cmd'] == 'Model' and text != 'Back':
            with open ('data/models.json') as f:
                models = json.load(f)
            for key, value in models.items():
                if text == key:
                    user_session['model'] = key
                    keyboard_dict["Default"]["message"] = f'Model has been set to {key}'
                    break        

        logger.info(f'keyboard_dict: {keyboard_dict}')

        # Update user session
        user_session['last_cmd'] = message['text']
        # Save user session
        panthera.save_user_session(message['from']['id'], user_session)

        return JSONResponse(content={
            "type": "keyboard",
            "body": keyboard_dict
            })

    else:
        # answer = panthera.llm_request(user_session, message)
        answer = 'llm_request'

    return JSONResponse(content={
        "type": "text",
        "body": str(answer)
        })
