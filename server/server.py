from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import json
from panthera import Panthera
import re
import pandas as pd
import matplotlib.pyplot as plt
# import base64

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


def keyboard_modificator(current_screen, user_session, menu, message):
    # Format message with current values if needed
    if '%s' in message:  
        if current_screen == 'Model':
            model = user_session['model']
        elif current_screen == 'Language':
            language = user_session['language']
        elif current_screen == 'Topic' or current_screen == 'Reports':
            if 'topic' in user_session:
                topic = user_session['topic']
            else:
                topic = 'None'
        message = message % model if 'model' in locals() else message
        message = message % language if 'language' in locals() else message
        message = message % topic if 'topic' in locals() else message
        menu[current_screen]['message'] = message


def get_keyboard(user_session, current_screen):

    with open('data/menu.json') as f:
        menu = json.load(f)

    if current_screen in menu:
        message = menu[current_screen]['message']
        keyboard_modificator(current_screen, user_session, menu, message)        
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
    message_type = panthera.get_message_type(user_session, text)
    logger.info(f'message_type: {message_type}')

    system_content = None

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

        if message['text'] != 'Back':
            # Model
            if user_session['last_cmd'] == 'Model':
                logger.info(f'Button last_cmd: Model. text is: {text}')
                with open ('data/models.json') as f:
                    models = json.load(f)
                for key, value in models.items():
                    if text == key:
                        user_session['model'] = key
                        keyboard_dict["message"] = f'Model has been set to {key}'
                        break
            # Language
            elif user_session['last_cmd'] == 'Language':
                logger.info(f'Button last_cmd: Language. text is: {text}')
                with open ('data/languages.json') as f:
                    languages = json.load(f)
                for key, value in languages.items():
                    if text == key:
                        user_session['language'] = key
                        keyboard_dict["message"] = f'Language has been set to {key}'
                        break
            # Topic
            elif user_session['last_cmd'] == 'Topic':
                logger.info(f'Button last_cmd: Topic. text is: {text}')
                with open ('data/topics.json') as f:
                    topics = json.load(f)
                for key, value in topics.items():
                    if text == key:
                        user_session['topic'] = key
                        panthera.reset_chat(message['chat']['id'])
                        system_content = value['system']
                        assistant_message = value['assistant']
                        
                        # Log assistant's message
                        bot_message = panthera.default_bot_message(
                            message,
                            assistant_message
                            )
                        # Log message
                        panthera.log_message(bot_message)

                        keyboard_dict["message"] = f'Topic has been set to {key}\n{assistant_message}'
                        break
            # Report
            elif user_session['last_cmd'] == 'Reports' and text == 'Progress report':
                logger.info(f'Button last_cmd: Reports. text is: {text}')
                # Convert to pandas DataFrame
                topic = user_session['topic']
                evaluations = user_session['topics'][topic]['evaluations']
                # Creating a DataFrame
                df = pd.DataFrame(evaluations)
                df['date'] = pd.to_datetime(df['date'], unit='s')  # Converting Unix timestamp to datetime
                df['topic'] = topic  # Adding topic as a column
                # Set the x-axis to only include the dates we have data for
                plt.figure(figsize=(10, 6))
                # Calculate the width of each bar dynamically based on the number of evaluations
                # This is to ensure that bars don't merge into each other
                # We divide by the number of evaluations to ensure that the total width of all bars is less than 1
                bar_width = (df['date'].max() - df['date'].min()) / len(df) / pd.Timedelta(days=1)
                # Plot the bars with a fixed width
                plt.bar(df['date'], df['value'], color='lightblue', width=bar_width)
                # Set x-axis ticks to be exactly the dates from the dataset
                plt.xticks(df['date'])
                # Rotate the x-axis labels for better readability
                plt.xticks(rotation=70)
                plt.title(f"Progress of the Topic: {topic}")
                plt.xlabel('Date')
                plt.ylabel('Value')
                # Save to file with unique name
                # Create data/plots folder if it doesn't exist
                if not os.path.exists('data/plots'):
                    os.makedirs('data/plots')
                filename = f'data/plots/{chat_id}_evaluation_plot.png'
                plt.savefig(filename)
                # Close the plot
                plt.close()

                response = FileResponse(filename, media_type="image/png")

                # Return the image data
                """with open(filename, 'rb') as f:
                    image_data = f.read()"""
                logger.info(f'image_data filename: {filename}')
                # Remove the file
                # os.remove(filename)
                # Return the image data                
                # Encode image to base64 
                # image_data = base64.b64encode(image_data)
                """return JSONResponse(content={
                    "type": "image",
                    "body": image_data
                    })"""
                # return FileResponse(image_data, media_type="image/png")
                return response
            else:
                logger.info(f'Button has not reacted. last_cmd: {user_session["last_cmd"]}. text is: {text}')

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
        # Read the system_content from the topics by user_session['topic'] if it is set
        if 'topic' in user_session:
            with open ('data/topics.json') as f:
                topics = json.load(f)
            system_content = topics[user_session['topic']]['system']
        answer = panthera.llm_request(user_session, message, system_content=system_content)
        
        # Evaluation log: If [num] in the answer, extract the num and set the evaluation
        match = re.search(r'\[(10|[0-9])\]', answer)
        if match:
            num = match.group(1)
            panthera.add_evaluation_to_topic(
                user_session,
                topic_name=user_session['topic'],
                value=int(num)
            )
            # Save user session
            panthera.save_user_session(message['from']['id'], user_session)

    return JSONResponse(content={
        "type": "text",
        "body": str(answer)
        })
