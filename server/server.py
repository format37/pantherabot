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
from telebot.formatting import escape_markdown
# from telebot.types import InlineQueryResultPhoto
import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path

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

# def remove_unsupported_tags(text, supported_tags):
#     # Convert the supported tags string to a set for faster lookup
#     supported_tags_set = set(supported_tags.split('>'))
    
#     # Regular expression pattern to match HTML tags
#     tag_pattern = re.compile(r'<(/?\w+)(.*?)>')
    
#     # Function to replace unsupported tags
#     def replace_tag(match):
#         tag_name = match.group(1)
#         if '<' + tag_name + '>' in supported_tags_set:
#             return match.group(0)  # Keep the tag if it's supported
#         else:
#             return ''  # Remove the tag if it's not supported
    
#     # Replace unsupported tags using the replace_tag function
#     cleaned_text = tag_pattern.sub(replace_tag, text)
    
#     return cleaned_text

# def escape_markdown(text):
#     escape_chars = r'_*[]()~`>#+-=|{}.!'
#     return re.sub(r'([{}])'.format(re.escape(escape_chars)), r'\\\1', text)

# def prepare_markdown(text):
#     # Characters to be escaped
#     escape_chars = '_*[]()~`>#+-=|{}.!'

#     # Escape function
#     def escape(char):
#         return '\\' + char

#     # Escape the characters in the text
#     for char in escape_chars:
#         text = text.replace(char, escape(char))

#     # Handle special cases

#     # Escape '\' character
#     text = text.replace('\\', '\\\\')

#     # Escape '`' and '\' characters inside pre and code entities
#     text = text.replace('```', '\\`\\`\\`')
#     text = text.replace('`', '\\`')

#     # Escape ')' and '\' characters inside the (...) part of the inline link and custom emoji definition
#     def escape_link_emoji(match):
#         content = match.group(1)
#         content = content.replace(')', '\\)')
#         content = content.replace('\\', '\\\\')
#         return f'({content})'

#     text = re.sub(r'\(([^)]*)\)', escape_link_emoji, text)

#     # Handle ambiguity between italic and underline entities
#     text = text.replace('___', '___*')

#     return text

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

# async def call_llm_response(message, message_text, chat_id, reply):
async def call_llm_response(chat_id, message_id, message_text, reply):
    # chat_id = message['chat']['id']
    # if 'topic' in user_session:
            # with open ('data/topics.json') as f:
            #     topics = json.load(f)
            # system_content = topics[user_session['topic']]['system']
    # Log the current_date
    current_date = pd.Timestamp.now()
    # -3h from the current_date
    current_date = current_date - pd.Timedelta(hours=3)
    logger.info(f'current_date: {current_date}')
    # if reply:
        # answer = await panthera.llm_request(bot, message, message_text)
    answer = await panthera.llm_request(chat_id, message_id, message_text)

    if answer == '':
        return JSONResponse(content={
        "type": "empty",
        "body": ''
        })
    
    if len(answer) > 4096:
        try:
            filename = await panthera.generate_filename(answer)
        except Exception as e:
            logger.info(f"Error generating filename: {e}")
            filename = "response.txt"
        if not filename.endswith(".txt"):
            logger.info(f"Filename [{filename}] does not end with '.txt'. Appending '.txt'...")
            filename += ".txt"
        if len(filename) > 64:
            logger.info(f"Filename [{filename}] is too long. Truncating...")
            filename = "response.txt"
        # Create in-memory file-like object
        buffer = BytesIO(answer.encode())
        # buffer.name = 'response.txt'  # Give a name to the file
        buffer.name = filename  # Give a name to the file
        buffer.seek(0)  # Move to the beginning of the BytesIO buffer
        bot.send_document(chat_id, buffer, reply_to_message_id=message_id)
        return JSONResponse(content={
            "type": "empty",
            "body": ''
        })
    
    formatting = {
        "&&&": "u447a0a7930e94a888a86a9ee09042458",
        "@@@": "u4cf178c998d04dfb88897ac3e49630bf",
        "%%%": "u9604214d2ab14a539623d63f4a3b7e3b",
        "~~~": "u06f4b328e72240c8b2909652a70af831",
        "||": "u955ba36d498a48119ac522100978f861",
        "```": "u795fe7bde93a4aaf9351a2064b1ab484"
    }
    for key, value in formatting.items():
        answer = answer.replace(key, value)
    answer = escape_markdown(answer)
    for key, value in formatting.items():
        answer = answer.replace(value, key)
    answer = answer.replace('&&&', '*') # bold
    answer = answer.replace('%%%', '_') # italic
    answer = answer.replace('@@@', '__') # underline
    answer = answer.replace('~~~', '~') # strikethrough
    try:
        logger.info(f'### sending MarkdownV2: {answer}')
        if reply:
            # bot.send_message(chat_id, answer, reply_to_message_id=message['message_id'], parse_mode='MarkdownV2')
            bot.send_message(chat_id, answer, reply_to_message_id=message_id, parse_mode='MarkdownV2')
        else:
            bot.send_message(chat_id, answer, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f'Error sending markdown: {e}')
        answer = escape_markdown(answer)
        logger.info(f'### sending escaped: {answer}')
        if reply:
            # bot.send_message(chat_id, answer, reply_to_message_id=message['message_id'], parse_mode='MarkdownV2')
            bot.send_message(chat_id, answer, reply_to_message_id=message_id, parse_mode='MarkdownV2')
        else:
            bot.send_message(chat_id, answer, parse_mode='MarkdownV2')

    return JSONResponse(content={
        "type": "empty",
        "body": ''
        })

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info('call_message')
    
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
        logger.info('No text, caption, photo or document in the message')
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })
    
    # Preparing text
    if 'text' in message:
        text = message['text']
    elif 'caption' in message:
        text = message['caption']
    else:
        text = ''

    # Skip processing if text starts with "response:"
    if text.startswith('response:'):
        return JSONResponse(content={
            "type": "empty",
            "body": ''
        })

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

    # Help command
    elif text == '/help':
        logger.info('Help CMD')
        help_text = """🤖 *Bot Features*

*Basic Interaction*
• Chat naturally in private messages
• Use /\* or /\. prefix in group chats
• Reply to my messages to continue conversation

*Smart Tools*
• Python code execution & debugging
• Google search with links
• YouTube video search
• Wikipedia lookups
• Wolfram Alpha calculations
• Image understanding & analysis
• Image generation with Flux Pro 1\.1

*Memory & Context*
• Maintains conversation history
• /reset \- Clear chat memory
• /update\_prompt \<text\> \- Update system prompt
• /reset\_prompt \- Reset to default prompt

*File Handling*
• Reads text & JSON files
• Generates detailed responses
• Auto\-splits long responses into files

*Group Chat Features*
• @gptaidbot \- Quote bot's last pm message
• @gptaidbot photo \- Quote my last image
• Authorized group access control

*Admin Commands*
• /add \<user\_id\> \- Add user access
• /remove \<user\_id\> \- Remove user access
• @gptaidbot \*\*\* \- Select a group to send bot's thoughts"""

        bot.send_message(message['chat']['id'], help_text, parse_mode='MarkdownV2')
        return JSONResponse(content={
            "type": "empty",
            "body": ''
        })

    answer = 'empty'

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
        elif text.startswith('response:'):
            # example: text == "response:-888407449"
            chat_id = text.split(':')[1]
            # user_session = panthera.get_user_session(user_id)
            message_text = ""
            # await call_llm_response(message, message_text, chat_id, False)
            await call_llm_response(chat_id, message["message_id"], message_text, False)

        
    chat_id = message['chat']['id']

    user_session = panthera.get_user_session(message['from']['id'])
    logger.info(f'user_session: {user_session}')
    
    message_type = panthera.get_message_type(user_session, text)
    logger.info(f'message_type: {message_type}')

    system_content = None

    # if message text is /start
    if text == '/start':
        answer = """Hi. I am Janet, conversational GPT bot with langchain tools:
* Python
* Google search
* Youtube search
* Wolfram alpha
* Wikipedia search
* Image understanding
* Flux 1.1 pro image generator
* Reading text & json files
* Updating system prompt
* Reseting system prompt
* Asking reasoning o1-preview expert

Commands:
/* or ./ prefix in a group chat to call me.
"@gptaidbot" to cite my last personal message in a group chat.
"@gptaidbot photo" to cite my last photo from personal message in a group chat.
"""
# @gptaidbot *** in other group to choose a group where to send the message.

        bot.send_message(chat_id, answer)
        # return empty
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })
    
    # Extractinf file list from the message
    file_list = ''
    if 'photo' in message or 'document' in message:
        file_list = panthera.get_message_file_list(bot, message)
    
    # Save message to the Chat history
    first_name = panthera.get_first_name(message)
    # if 'first_name' in message['chat']:
    #     first_name = message['from']['first_name']
    # else:
    #     first_name = message['from']['username']
    # panthera.log_message(message)
    message_date = message['date']
    # Convert 'date': 1718167018 to '2024-06-06 12:36:58'
    message_date = pd.to_datetime(message_date, unit='s')
    time_passed = pd.Timestamp.now() - message_date
    message_text = f"user_name: {first_name}"
    message_text += f"\nchat_id: {chat_id}"
    message_text += f"\nmessage_id: {message['message_id']}"
    if "reply_to_message" in message:
        message_text += f"\nreply_to_message: {message['reply_to_message']['message_id']}"
    message_text += f"\nmessage_date: {message_date}"
    if file_list != '':
        message_text += f"\nfile_list: {file_list}"
    if text != '':
        message_text += f"\nmessage_text: {text}"
    panthera.save_to_chat_history(
        chat_id,
        f"{message_text}",
        message["message_id"],
        "HumanMessage"
    )

    if text == '':
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })

    if message['chat']['type'] == 'private' \
        or text.startswith('/*') \
        or text.startswith('/.') \
        or panthera.is_reply_to_ai_message(message):
        # await call_llm_response(message, message_text, message['chat']['id'], True)
        await call_llm_response(chat_id, message["message_id"], message_text, True)
        
    return JSONResponse(content={
        "type": "empty",
        "body": ''
        })

def get_group_name(chat_id):
    try:
        chat = bot.get_chat(chat_id)
        return chat.title  # This will return the name of the group
    except Exception as e:
        return str(e)  # Handle exceptions, e.g., invalid chat_id

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
    inline_query_id = message['inline_query_id']
    
    query = message.get('query', '').lower()
    user_id = message['from_user_id']

    if query.endswith('photo'):
        image_dir = f"data/users/{user_id}/images"
        if not os.path.exists(image_dir):
            logger.info(f"No images found for user {user_id}")
            return JSONResponse(content={"status": "ok"})

        image_files = os.listdir(image_dir)
        if not image_files:
            logger.info(f"No images found in {image_dir}")
            return JSONResponse(content={"status": "ok"})

        inline_elements = []
        image_files = os.listdir(image_dir)

        # Sort files by creation time, newest first
        sorted_files = sorted(image_files, 
                            key=lambda x: os.path.getctime(os.path.join(image_dir, x)), 
                            reverse=True)
        file_number = 0
        for idx, image_file in enumerate(sorted_files):
            image_path = os.path.join(image_dir, image_file)
            file_number += 1
            # Remove all files that have more than 6
            if file_number > 6:
                os.remove(image_path)
                logger.info(f"[-] image_file: {image_file}")
            else:                
                logger.info(f"[+] image_file: {image_file}")
                uid = hashlib.md5(image_file.encode()).hexdigest()
                element = telebot.types.InlineQueryResultCachedPhoto(
                    id = uid,
                    photo_file_id = image_file
                )
                inline_elements.append(element)

        bot.answer_inline_query(
            inline_query_id,
            inline_elements,
            cache_time=0,
            is_personal=True
        )
        return JSONResponse(content={"status": "ok"})
    elif query.endswith('***'):
        # Check if user is in admins.txt
        admins = []
        with open('data/admins.txt', 'r') as f:
            admins = f.read().splitlines()
        
        if str(user_id) not in admins:
            logger.info(f"User {user_id} is not an admin")
            return JSONResponse(content={"status": "ok"})
        # # There the LLM is answering what they think without prompt
        # user_session = panthera.get_user_session(user_id)
        # message_text = ""
        # # await call_llm_response(user_session, message, message_text)
        # logger.info(f"*** message: {message}")
        # logger.info(f"*** authorization: {authorization}")
        # chat_id = "-888407449"
        # # group_name = get_group_name(chat_id)
        # chat = bot.get_chat(chat_id)
        # logger.info(f"*** chat.title: {chat.title}")
        # logger.info(f"*** chat.type: {chat.type}")

        # Iterate all possible chats that bot participate in
        chats_folder = 'data/chats/'
        # Read list of folders in chats_folder
        folders = os.listdir(chats_folder)
        
        inline_elements = []
        folder_id = 0
        # Iterate all folders
        for folder in folders:
            chat = bot.get_chat(folder)
            if chat.type == 'group':
                logger.info(f"*** chat.title: {chat.title}")
                element = telebot.types.InlineQueryResultArticle(
                    folder_id,
                    chat.title,
                    telebot.types.InputTextMessageContent(f"response:{folder}"),
                )
                inline_elements.append(element)
                folder_id += 1
            else:
                logger.info(f"*** not included chat.type: {chat.type}")

        bot.answer_inline_query(
                inline_query_id,
                inline_elements,
                cache_time=0,
                is_personal=True
            )
        return JSONResponse(content={"status": "ok"})

    else:
        # Check is path ./data/{user_id}/ exists. If not, return 'no data'
        # data_folder = f"data/chats/{message['from_user_id']}/"
        data_folder = f"data/users/{user_id}/chats/{message['from_user_id']}/"
        if not os.path.exists(data_folder):
            logger.info(f"Folder is not exist: {data_folder}")
            return JSONResponse(content={"status": "ok"})
        # Is path ./data/{user_id}/ have files. If not, return 'no data'
        files = os.listdir(data_folder)
        if not files:
            logger.info(f"Folder is empty: {data_folder}")
            return JSONResponse(content={"status": "ok"})
        # Reads the latest file, sorted by name
        files.sort()
        # Latest file is json. Load and read the message['text']
        with open(data_folder + files[-1]) as f:
            data = json.load(f)
        # Returns the file content
        logger.info(f"inline data: {data}")
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
