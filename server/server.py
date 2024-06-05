from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import json
from panthera import Panthera
import re
import pandas as pd
import matplotlib.pyplot as plt
# from telebot import TeleBot
import telebot

# Initialize FastAPI
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

server_api_uri = 'http://localhost:8081/bot{0}/{1}'
telebot.apihelper.API_URL = server_api_uri
logger.info(f'Setting API_URL: {server_api_uri}')

server_file_url = 'http://localhost:8081'
telebot.apihelper.FILE_URL = server_file_url
logger.info(f'Setting FILE_URL: {server_file_url}')

with open('config.json') as config_file:
    bot = telebot.TeleBot(json.load(config_file)['TOKEN'])

panthera = Panthera()


@app.get("/test")
async def call_test():
    logger.info('call_test')
    return JSONResponse(content={"status": "ok"})


def keyboard_modificator(current_screen, user_session, menu, message):
    # Format message with current values if needed
    """if '%s' in message:  
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
        menu[current_screen]['message'] = message"""


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


def user_access(message):
    # # Initialize the bot
    # bot = TeleBot(token)
    # Get list of users from ./data/users.txt
    with open('data/users.txt') as f:
        users = f.read().splitlines()
    # Check if user is in the list
    if str(message['from']['id']) in users:
        return True
    # If chat is not private
    elif message['chat']['type'] != 'private':
        # Create folder ./data/granted_groups if it doesn't exist
        if not os.path.exists('data/granted_groups'):
            os.makedirs('data/granted_groups')
        # Check if group is in the ./data/granted_groups/<chat_id>.txt
        if os.path.exists(f'data/granted_groups/{message["chat"]["id"]}.txt'):
            return True
        # Create folder ./data/denied_groups if it doesn't exist
        if not os.path.exists('data/denied_groups'):
            os.makedirs('data/denied_groups')
        # Check if group is in the ./data/denied_groups/<chat_id>.txt
        if os.path.exists(f'data/denied_groups/{message["chat"]["id"]}.txt'):
            return False
        # Utilize get_chat_member to check is user from list in group
        for user in users:
            # Get chat member
            try:
                member = bot.get_chat_member(message['chat']['id'], user)
                if member.status in ["member", "administrator", "creator"]:
                    # logger.info(f'user_access: {user} is in the {message["chat"]["id"]} group with status {member.status}')
                    # Write group to the ./data/granted_groups/<chat_id>.txt
                    with open(f'data/granted_groups/{message["chat"]["id"]}.txt', 'w') as f:
                        f.write(str(message["chat"]["id"]))
                    return True
            except Exception as e:
                # logger.info(f'user_access: {user} is not in the {message["chat"]["id"]} group')
                pass
        # logger.info(f'user_access: {message["from"]["id"]} is not in the {message["chat"]["id"]} group')
    else:
        # logger.info(f'user_access: {message["from"]["id"]} is not in the users list')
        pass
    
    # Write group to the ./data/denied_groups/<chat_id>.txt
    if not os.path.exists('data/denied_groups'):
        os.makedirs('data/denied_groups')
    with open(f'data/denied_groups/{message["chat"]["id"]}.txt', 'w') as f:
        f.write(str(message["chat"]["id"]))
    
    return False

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info('call_message')

    # token = None
    # if authorization and authorization.startswith("Bearer "):
    #     token = authorization.split(" ")[1]
    
    # if token:
    #     logger.info(f'Bot token: {token}')
    #     pass
    # else:
    #     answer = 'Bot token not found. Please contact the administrator.'
    #     return JSONResponse(content={
    #         "type": "text",
    #         "body": str(answer)
    #     })
    
    message = await request.json()
    logger.info(message)

    if not user_access(message):
        if message['chat']['type'] == 'private':
            answer = "You are not authorized to use this bot.\n"
            answer += "Please forward this message to the administrator.\n"
            answer += f'User id: {message["from"]["id"]}'
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
            })
        else:
            return JSONResponse(content={
                "type": "empty",
                "body": ''
            })

    if  not 'text'      in message and \
        not 'caption'   in message and \
        not 'photo'     in message and \
        not 'document'  in message:
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })
    
    if 'text' in message:
        text = message['text']
    elif 'caption' in message:
        text = message['caption']
    else:
        text = ''
    
    data_path = 'data/'
    # Read user_list from ./data/users.txt
    with open(data_path + 'users.txt', 'r') as f:
        user_list = f.read().splitlines()

    # Add user CMD
    if text.startswith('/add'):
        logger.info(f'Add user CMD: {text}')
        # Check is current user in atdmins.txt
        admins = []
        with open(data_path + 'admins.txt', 'r') as f:
            admins = f.read().splitlines()
        if str(message['from']['id']) not in admins:
            answer = "You are not authorized to use this command."
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        # split cmd from format /add <user_id>
        cmd = text.split(' ')
        if len(cmd) != 2:
            answer = "Invalid command format. Please use /add <user_id>."
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        # add user_id to user_list
        user_id = cmd[1]
        user_list.append(user_id)
        # write user_list to ./data/users.txt
        with open(data_path + 'users.txt', 'w') as f:
            f.write('\n'.join(user_list))
        answer = f'User {user_id} added successfully.'
        return JSONResponse(content={
            "type": "text",
            "body": str(answer)
            })

    # Remove user CMD
    elif text.startswith('/remove'):
        logger.info(f'Remove user CMD: {text}')
        # Check is current user in atdmins.txt
        admins = []
        with open(data_path + 'admins.txt', 'r') as f:
            admins = f.read().splitlines()
        if str(message['from']['id']) not in admins:
            answer = "You are not authorized to use this command."
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        # split cmd from format /remove <user_id>
        cmd = text.split(' ')
        if len(cmd) != 2:
            answer = "Invalid command format. Please use /remove <user_id>."
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        # remove user_id from user_list
        user_id = cmd[1]
        user_list.remove(user_id)
        # write user_list to ./data/users.txt
        with open(data_path + 'users.txt', 'w') as f:
            f.write('\n'.join(user_list))
        answer = f'User {user_id} removed successfully.'
        return JSONResponse(content={
            "type": "text",
            "body": str(answer)
            })

    answer = 'empty'

    

    # if message text is /reset
    # if chat type private
    if 'text' in message:
        if text == '/reset' and message['chat']['type'] == 'private':
            panthera.reset_chat(message['chat']['id'])
            answer = 'Chat messages memory has been cleaned'
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        # if chat type not private
        elif text.startswith('/reset@') and message['chat']['type'] != 'private':
            panthera.reset_chat(message['chat']['id'])
            answer = 'Chat messages memory has been cleaned'
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        
    chat_id = message['chat']['id']
    
    # if 'first_name' in message['chat']:
    #     first_name = message['from']['first_name']
    # else:
    #     first_name = message['from']['username']
    # panthera.log_message(message)
    # panthera.save_to_chat_history(
    #     chat_id,
    #     f"{first_name}: {text}",
    #     message["message_id"],
    #     "HumanMessage"
    # )
    # 

    user_session = panthera.get_user_session(message['from']['id'])
    logger.info(f'user_session: {user_session}')
    
    message_type = panthera.get_message_type(user_session, text)
    logger.info(f'message_type: {message_type}')

    system_content = None

    # if message text is /start
    if text == '/start':
        answer = 'Welcome to the conversational gpt bot.\nPlease, send me a regular message in private chat, or use /* prefix in a group chat to call me.'

        # elif message['text'] == '/configure': # TODO: account the non-private chats
        # elif user_session['last_cmd'] != 'start':
    
        # elif message_type == 'button':
        
        keyboard_dict = get_keyboard(user_session, text)

        if text != 'Back':
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
                #with open(filename, 'rb') as f:
                    #image_data = f.read()
                logger.info(f'image_data filename: {filename}')
                # Remove the file
                # os.remove(filename)
                # Return the image data                
                # Encode image to base64 
                # image_data = base64.b64encode(image_data)
                # return FileResponse(image_data, media_type="image/png")
                return response
            else:
                logger.info(f'Button has not reacted. last_cmd: {user_session["last_cmd"]}. text is: {text}')

        logger.info(f'keyboard_dict: {keyboard_dict}')

        # Update user session
        user_session['last_cmd'] = text
        # Save user session
        panthera.save_user_session(message['from']['id'], user_session)

        return JSONResponse(content={
            "type": "keyboard",
            "body": keyboard_dict
            })

    elif message['chat']['type'] == 'private' \
        or text.startswith('/*') \
        or text.startswith('/.'):
        # Read the system_content from the topics by user_session['topic'] if it is set
        if 'topic' in user_session:
            with open ('data/topics.json') as f:
                topics = json.load(f)
            system_content = topics[user_session['topic']]['system']
        answer = panthera.llm_request(bot, user_session, message, system_content=system_content)
        
        # Evaluation log: If [num] in the answer, extract the num and set the evaluation
        """match = re.search(r'\[(10|[0-9])\]', answer)
        if match:
            num = match.group(1)
            panthera.add_evaluation_to_topic(
                user_session,
                topic_name=user_session['topic'],
                value=int(num)
            )
            # Save user session
            panthera.save_user_session(message['from']['id'], user_session)"""
    else:
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })

    return JSONResponse(content={
        "type": "text",
        "body": str(answer)
        })
  
# Post inline query
@app.post("/inline")
async def call_inline(request: Request, authorization: str = Header(None)):
    logger.info('call_inline')

    """This function:
    1. Check is path ./data/{['from_user']['id']}/ exists. If not, return 'no data'
    2. Is path ./data/{['from_user']['id']}/ have files. If not, return 'no data'
    3. Reads the latest file, sorted by name
    4. Returns the file content
    """
    message = await request.json()
    logger.info(f'inline content: {message}')
    # from_user_id = message['from_user_id']
    inline_query_id = message['inline_query_id']
    # expression = message['query']
    # message = content['inline_query']
    # Check is path ./data/{user_id}/ exists. If not, return 'no data'
    data_folder = f"data/chats/{message['from_user_id']}/"
    if not os.path.exists(data_folder):
        logger.info(f"Folder is not exist: {data_folder}")
        # return JSONResponse(content={
        #     "title": "no data",
        #     "message_text": "no data"
        #     })
        return JSONResponse(content={"status": "ok"})
    # Is path ./data/{user_id}/ have files. If not, return 'no data'
    files = os.listdir(data_folder)
    if not files:
        logger.info(f"Folder is empty: {data_folder}")
        # return JSONResponse(content={
        #     "title": "no data",
        #     "message_text": "no data"
        #     })
        return JSONResponse(content={"status": "ok"})
    # Reads the latest file, sorted by name
    files.sort()
    # Latest file is json. Load and read the message['text']
    with open(data_folder + files[-1]) as f:
        data = json.load(f)
    # Returns the file content
    logger.info(f"inline data: {data}")
    # return JSONResponse(content={
    #         "type": "inline",
    #         "body": [data['text']]
    #         })
    inline_elements = []
    element = telebot.types.InlineQueryResultArticle(
        0,
        data['text'],
        telebot.types.InputTextMessageContent(data['text']),
    )
    inline_elements.append(element)

    bot.answer_inline_query(
            inline_query_id,
            inline_elements,
            cache_time=0,
            is_personal=True
        )
    return JSONResponse(content={"status": "ok"})