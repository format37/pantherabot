from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import json
from panthera import Panthera, record_message_id, record_raw_text
import edits
import memory
import tools_cli
import re
import time
import pandas as pd
# from telebot import TeleBot
import telebot
from telebot.formatting import escape_markdown
import hashlib
from io import BytesIO
import asyncio
import requests

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

# Media group buffer for handling Telegram albums
# Structure: {media_group_id: {"items": {message_id: {"caption": str, "files": [(path, file_unique_id)]}},
#             "chat_id": int, "chat_type": str, "message": dict, "task": asyncio.Task}}
media_group_buffers = {}
# How long an album waits for its next item before it is saved and answered.
MEDIA_GROUP_WAIT_SECONDS = 2


def attached_files(message):
    """[(path, file_unique_id)] of the files a message brings into the history.

    The message's own photo or readable document first, then the one in the
    message it replies to.
    """
    files = []
    sources = [message]
    if 'reply_to_message' in message:
        sources.append(message['reply_to_message'])
    for source in sources:
        attached = panthera.attached_file(source)
        if attached is None:
            continue
        # Telegram local server returns absolute paths like /6014837471:AAE5.../photos/file.jpg
        # Strip the /{BOT_ID}: prefix so the path matches the container volume mount target
        for raw_path in panthera.get_message_file_list(bot, source):
            clean_path = re.sub(r'^/[^/]+:', '/', raw_path)
            if all(clean_path != path for path, _ in files):
                files.append((clean_path, attached[1]))
                logger.info(f"Image path: {clean_path}")
    return files


def human_record(message, message_id, text, files, edit_date=None):
    """The history record of a human message.

    One builder for /message, albums and /edited_message, so that an edit
    rewrites a record the way it was first written. `message_id` is the id the
    record is filed under (an album's first item), `files` is
    [(path, file_unique_id)], and `edit_date` is Telegram's unix time.
    """
    image_paths = [path for path, _ in files]
    message_text = f"user_name: {panthera.get_first_name(message)}"
    message_text += f"\nchat_id: {message['chat']['id']}"
    message_text += f"\nmessage_id: {message_id}"
    if "reply_to_message" in message:
        message_text += f"\nreply_to_message: {message['reply_to_message']['message_id']}"
    # Convert 'date': 1718167018 to '2024-06-06 12:36:58'
    message_text += f"\nmessage_date: {pd.to_datetime(message['date'], unit='s')}"
    if edit_date is not None:
        edit_date = str(pd.to_datetime(edit_date, unit='s'))
        message_text += f"\nedit_date: {edit_date}"
    if image_paths:
        message_text += f"\nfile_list: {image_paths}"
    if text != '':
        message_text += f"\nmessage_text: {text}"
    record = {
        "type": "HumanMessage",
        "text": message_text,
        "images": image_paths,
        "message_id": int(message_id),
        "raw_text": text,
        "file_unique_ids": [unique_id for _, unique_id in files],
    }
    if edit_date is not None:
        record["edit_date"] = edit_date
    return record


def album_text(captions):
    """An album's text: its items' distinct captions, in message order."""
    seen = []
    for caption in captions.values():
        if caption and caption not in seen:
            seen.append(caption)
    return '\n'.join(seen)


@app.get("/test")
async def call_test():
    logger.info('call_test')
    return JSONResponse(content={"status": "ok"})

def load_authorized_users():
    """User ids listed in data/users.txt."""
    with open('data/users.txt') as f:
        return f.read().splitlines()


def is_authorized_sender(message):
    """True when this particular sender is in data/users.txt.

    user_access() answers "may this chat be served at all" — in a granted group
    it says yes to every member. This answers "may this sender make the bot run
    tools", which is a different question: a guest gets the conversation, an
    authorized user gets the tools.
    """
    return str(message['from']['id']) in load_authorized_users()


def parse_command(text):
    """Split '/cmd@bot rest of line' into ('/cmd', 'rest of line')."""
    if not text.startswith('/'):
        return '', ''
    head, _, rest = text.partition(' ')
    return head.split('@', 1)[0], rest.strip()


def user_access(message):
    # # Initialize the bot
    # bot = TeleBot(token)
    # Get list of users from ./data/users.txt
    users = load_authorized_users()
    # Check if user is in the list
    if str(message['from']['id']) in users:
        return True
    # A chat id in users.txt (/add <chat_id>) opens that group to every member,
    # bots included. It grants the conversation only: tools still go by sender.
    elif str(message['chat']['id']) in users:
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

def send_rich_message(chat_id, markdown_text, reply_to=None):
    """Send a Telegram rich message (Markdown) via the local Bot API server.

    Rich messages support headings, tables, lists, block quotes, collapsible
    <details>, native LaTeX, etc. and allow up to 32768 chars. telebot has no
    wrapper for sendRichMessage yet, so we POST to the local server directly.
    Returns True on success, False on any failure so the caller can fall back
    to the legacy MarkdownV2 send path.
    """
    try:
        url = telebot.apihelper.API_URL.format(bot.token, 'sendRichMessage')
        payload = {
            "chat_id": chat_id,
            "rich_message": {"markdown": markdown_text},
        }
        if reply_to is not None:
            payload["reply_parameters"] = {
                "message_id": reply_to,
                "allow_sending_without_reply": True,
            }
        resp = requests.post(url, json=payload, timeout=30)
        data = resp.json()
        if data.get("ok"):
            return True
        logger.error(f"sendRichMessage rejected: {data}")
        return False
    except Exception as e:
        logger.error(f"sendRichMessage error: {e}")
        return False


def reply_parameters(reply_to):
    """A reply to `reply_to` that is sent even if that message is gone; None for no reply."""
    if reply_to is None:
        return None
    return telebot.types.ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)


# The local Bot API server answers an upload only once Telegram has the file,
# and 45 MB can take longer than telebot's default 30 s.
UPLOAD_TIMEOUT_SECONDS = 300


def send_outgoing(chat_id, reply_to, item):
    """Send one file a tool produced (bot_tools.Outgoing)."""
    def payload():
        buffer = BytesIO(item.data)
        buffer.name = item.filename
        return buffer

    if item.kind == 'photo':
        try:
            sent = bot.send_photo(
                int(chat_id), payload(), caption=item.caption, parse_mode=item.parse_mode,
                reply_parameters=reply_parameters(reply_to), timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        except telebot.apihelper.ApiTelegramException as e:
            # The tool can no longer report it, so a photo Telegram refuses
            # (size, proportions) gets a second chance as a document. Only a
            # refusal: after a timeout the photo may well have arrived.
            logger.error(f'Telegram refused {item.filename} for chat {chat_id} as a photo, '
                         f'sending it as a document: {e}')
        except Exception as e:
            logger.error(f'Could not send {item.filename} to chat {chat_id}: {e}')
            return
        else:
            if item.cache_inline:
                try:
                    tools_cli.remember_inline_photo(chat_id, sent.photo[-1].file_id)
                except Exception as e:
                    logger.error(f'Could not keep the photo for inline queries: {e}')
            return
    try:
        bot.send_document(
            int(chat_id), payload(), caption=item.caption, parse_mode=item.parse_mode,
            visible_file_name=item.filename, reply_parameters=reply_parameters(reply_to),
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.error(f'Could not send {item.filename} to chat {chat_id}: {e}')


def send_answer(chat_id, answer, reply_to, filename):
    """Send an answer's text: a rich message, or else a .txt document or MarkdownV2."""
    # Primary path: Telegram rich message (standard Markdown, up to 32768 chars).
    # Janet may emit headings, tables, lists, quotes, <details>, native LaTeX, etc.
    if len(answer) <= 32768 and send_rich_message(chat_id, answer, reply_to=reply_to):
        return

    # Fallback below: the rich send failed, or the response is too large to render
    # as a single message. A response over 4096 chars cannot fit a regular
    # sendMessage, so it is delivered as a .txt document instead.
    if len(answer) > 4096:
        # Create in-memory file-like object
        buffer = BytesIO(answer.encode())
        buffer.name = filename  # Give a name to the file
        buffer.seek(0)  # Move to the beginning of the BytesIO buffer
        bot.send_document(chat_id, buffer, reply_parameters=reply_parameters(reply_to))
        return

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
        bot.send_message(chat_id, answer, parse_mode='MarkdownV2',
                         reply_parameters=reply_parameters(reply_to))
    except Exception as e:
        logger.error(f'Error sending markdown: {e}')
        answer = escape_markdown(answer)
        logger.info(f'### sending escaped: {answer}')
        bot.send_message(chat_id, answer, parse_mode='MarkdownV2',
                         reply_parameters=reply_parameters(reply_to))


async def answer_filename(answer):
    """A .txt file name for an answer too long for a message."""
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
    return filename


def deliver(chat_id, reply_to, answer, outbox, filename):
    """Send an answer that passed the pre-send check: its files first, then its text.

    Runs in a worker thread, so a large upload does not stall the other chats.
    The generation is committed by then: an edit that arrives meanwhile
    changes the history only.
    """
    for item in outbox:
        send_outgoing(chat_id, reply_to, item)
    if answer:
        send_answer(chat_id, answer, reply_to, filename)


async def call_llm_response(chat_id, message_id, message_text, reply, tools_enabled=True,
                            from_history=True):
    """Answer a message, regenerating while the history it was given is edited.

    See edits.py. Each attempt reads the history and generates; files its
    tools produce wait in the attempt's outbox. Only the attempt that passes
    the pre-send check is saved and sent.
    """
    async def attempt(generation):
        system_prompt, user_prompt, context = panthera.prepare_prompt(
            chat_id, message_id, message_text, tools_enabled=tools_enabled,
            from_history=from_history,
        )
        generation.context = set(context)
        outbox = []
        answer = await panthera.generate(
            system_prompt, user_prompt, chat_id, message_id,
            tools_enabled=tools_enabled, outbox=outbox,
        )
        return answer, outbox

    async def commit(result):
        answer, outbox = result
        # The history keeps the answer that is actually sent, and only that one.
        if answer:
            panthera.save_to_chat_history(chat_id, answer, message_id, 'AIMessage')
        filename = await answer_filename(answer) if len(answer) > 4096 else None
        await asyncio.to_thread(
            deliver, chat_id, message_id if reply else None, answer, outbox, filename
        )

    await edits.answer(chat_id, message_id, attempt, commit)


# Answers run as tasks of their own. The request that brought a message
# returns once the message is saved, so the relay's worker threads stay free to
# forward edits while answers are generated or wait for a slot (edits.py).
answer_tasks = {}


def answer_in_background(chat_id, message_id, message_text, reply, **kwargs):
    """Start answering a message; do not wait for the answer."""
    task = asyncio.create_task(
        call_llm_response(chat_id, message_id, message_text, reply, **kwargs)
    )
    answer_tasks[task] = (chat_id, message_id)
    task.add_done_callback(_answer_finished)


def _answer_finished(task):
    chat_id, message_id = answer_tasks.pop(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error(f'Answering message {message_id} in chat {chat_id} failed',
                     exc_info=task.exception())


async def flush_media_group(media_group_id: str):
    """
    Flush accumulated media group messages after a timeout.
    Combines all images and text from the media group and processes them together.
    """
    # Wait for all images to arrive
    await asyncio.sleep(MEDIA_GROUP_WAIT_SECONDS)

    # Check if this media group still exists in buffer (could have been cancelled)
    if media_group_id not in media_group_buffers:
        logger.info(f"Media group {media_group_id} already processed or cancelled")
        return

    # Get the buffered data and remove it from the buffer. No await from here to
    # the save below: an edit of an item finds either the buffer or the record.
    buffer_data = media_group_buffers.pop(media_group_id)
    chat_id = buffer_data['chat_id']
    chat_type = buffer_data['chat_type']
    original_message = buffer_data['message']
    tools_enabled = is_authorized_sender(original_message)

    # Items reach us in any order (the relay forwards them from several
    # threads), so the album is filed under its first item, and its images and
    # captions are taken in message order.
    items = buffer_data['items']
    order = sorted(items)
    message_id = order[0]
    captions = {str(mid): items[mid]['caption'] for mid in order}
    text = album_text(captions)
    files = []
    for mid in order:
        for path, unique_id in items[mid]['files']:
            if all(path != known for known, _ in files):
                files.append((path, unique_id))

    logger.info(f"Flushing media group {media_group_id} with {len(files)} images")

    # Save to chat history with all images
    record = human_record(original_message, message_id, text, files)
    record['media_group_id'] = media_group_id
    record['captions'] = captions
    panthera.save_record(chat_id, message_id, record)
    message_text = record['text']

    # Process the complete media group only if conditions are met
    # (same conditions as in main message handler: private chat, prefix, or reply to bot)
    # The prefix may be on any item's caption.
    if chat_type == 'private' \
        or any(caption.startswith(('/*', '/.')) for caption in captions.values()) \
        or panthera.is_reply_to_ai_message(original_message):
        answer_in_background(chat_id, message_id, message_text, True,
                             tools_enabled=tools_enabled)

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

    # The chat is served (user_access above). Whether THIS sender may make the
    # bot execute anything is decided separately: in a granted group every member
    # can talk to the bot, but only users in data/users.txt get tools.
    tools_enabled = is_authorized_sender(message)
    logger.info(
        f"sender authorized: {'yes' if tools_enabled else 'no'} "
        f"(user {message['from']['id']}, chat {message['chat']['id']})"
    )

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

*Tools*
• Web search \(Perplexity\)
• Wolfram Alpha for math & science
• Image generation \(Gemini\)
• Image understanding & analysis
• Python code execution
• Math formula rendering

*Memory & Context*
• Maintains conversation history
• /reset \- Clear chat memory
• Ask me to remember something and it survives /reset
• /memory \- Show the notes saved in this chat
• /forget \<text\> \- Drop matching notes \(/forget all clears them\)

*Group Chat Features*
• @gptaidbot \- Quote bot's last pm message
• @gptaidbot photo \- Quote my last image
• @gptaidbot \*\*\* \- Select a group to send bot's thoughts

*Admin Commands*
• /add \<user\_id\> \- Add user access
• /remove \<user\_id\> \- Remove user access"""

        bot.send_message(message['chat']['id'], help_text, parse_mode='MarkdownV2')
        return JSONResponse(content={
            "type": "empty",
            "body": ''
        })

    answer = 'empty'

    if 'text' in message:
        command, command_args = parse_command(text)
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
        elif command == '/memory':
            # Long-term notes, deliberately untouched by /reset.
            try:
                content = memory.load(message['chat']['id'])
            except Exception as e:
                logger.error(f'/memory failed: {e}')
                content = ''
            if not content:
                answer = 'Memory is empty.'
            elif len(content) > 3500:
                answer = content[-3500:] + '\n\n(older notes not shown)'
            else:
                answer = content
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        elif command == '/forget':
            if not tools_enabled:
                answer = "Only authorized users can change this chat's memory."
            elif not command_args:
                answer = ('Usage: /forget <text> removes the notes containing that text, '
                          '/forget all clears the memory. /memory shows what is saved.')
            else:
                pattern = '*' if command_args.lower() in ('all', '*') else command_args
                try:
                    answer = memory.forget(message['chat']['id'], pattern)
                except Exception as e:
                    logger.error(f'/forget failed: {e}')
                    answer = f'Could not update memory: {e}'
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
                })
        elif text.startswith('response:'):
            logger.info(f"response: {text}")
            # example: text == "response:-888407449"
            chat_id = text.split(':')[1]
            
            # # Get the user's personal ID for reading their chat history
            # user_id = message['from']['id']
            # logger.info(f"Getting personal chat history for user_id: {user_id}")
            
            # # Read the user's personal chat history
            # panthera.read_chat_history(str(user_id))
            
            # # Concatenate chat history into a single text to guide the response
            # message_text = "Respond based on our previous conversation: "
            # for msg in panthera.chat_history:
            #     if isinstance(msg, HumanMessage):
            #         message_text += f"\nUser: {msg.content}"
            #     elif isinstance(msg, AIMessage):
            #         message_text += f"\nAssistant: {msg.content}"
            
            # # Limit the message_text to a reasonable size if needed
            # if len(message_text) > 4000:
            #     message_text = message_text[-4000:]
            
            # logger.info(f"Prepared message_text with personal chat history: {message_text[:100]}...")

            message_text = ''
            # The message_id is this private message's, not the group's: there is
            # no history record behind it.
            answer_in_background(chat_id, message["message_id"], message_text, False,
                                 tools_enabled=tools_enabled, from_history=False)
            return JSONResponse(content={
                "type": "empty",
                "body": ''
            })

        
    chat_id = message['chat']['id']

    user_session = panthera.get_user_session(message['from']['id'])
    logger.info(f'user_session: {user_session}')

    # if message text is /start
    if text == '/start':
        answer = """Hi. I am Janet, your AI assistant.

Commands:
/reset — clear chat memory
/memory — show what I remember about this chat
/forget <text> — drop matching notes (/forget all clears them)
/* or /. prefix in a group chat to call me.
"@gptaidbot" to cite my last personal message in a group chat.
"@gptaidbot photo" to cite my last photo from personal message in a group chat.
"""
        bot.send_message(chat_id, answer)
        # return empty
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })
    
    # Extract file list from the message (and the one it replies to), mapped to mounted paths
    files = attached_files(message)

    # Handle media groups (Telegram albums)
    if 'media_group_id' in message:
        media_group_id = message['media_group_id']
        chat_id = message['chat']['id']

        logger.info(f"Media group detected: {media_group_id}")

        # Initialize or update buffer for this media group
        if media_group_id not in media_group_buffers:
            media_group_buffers[media_group_id] = {
                'items': {},
                'chat_id': chat_id,
                'chat_type': message['chat']['type'],  # Store chat type for response condition check
                'message': message,  # Store message for is_reply_to_ai_message check
                'task': None
            }

        # Each item keeps its own caption (usually only one of them has one) and images
        items = media_group_buffers[media_group_id]['items']
        items[message['message_id']] = {'caption': text, 'files': files}
        logger.info(f"Added item {message['message_id']} with {len(files)} image(s) to media group buffer. Items: {len(items)}")

        # Cancel existing flush task if any
        if media_group_buffers[media_group_id]['task'] is not None:
            media_group_buffers[media_group_id]['task'].cancel()
            logger.info(f"Cancelled previous flush task for media group {media_group_id}")

        # Create new flush task with 2-second timeout
        media_group_buffers[media_group_id]['task'] = asyncio.create_task(
            flush_media_group(media_group_id)
        )
        logger.info(f"Created new flush task for media group {media_group_id}")

        # Return immediately - don't save to history or call LLM yet
        return JSONResponse(content={
            "type": "empty",
            "body": ''
        })

    # Save message to the Chat history
    record = human_record(message, message["message_id"], text, files)
    panthera.save_record(chat_id, message["message_id"], record)
    message_text = record['text']

    if text == '':
        return JSONResponse(content={
            "type": "empty",
            "body": ''
            })

    if message['chat']['type'] == 'private' \
        or text.startswith('/*') \
        or text.startswith('/.') \
        or panthera.is_reply_to_ai_message(message):
        answer_in_background(chat_id, message["message_id"], message_text, True,
                             tools_enabled=tools_enabled)
        
    return JSONResponse(content={
        "type": "empty",
        "body": ''
        })


def attached_unique_ids(message):
    """The file_unique_ids attached_files() would pair up, without asking the Bot API."""
    unique_ids = []
    sources = [message]
    if 'reply_to_message' in message:
        sources.append(message['reply_to_message'])
    for source in sources:
        attached = panthera.attached_file(source)
        if attached is not None and attached[1] not in unique_ids:
            unique_ids.append(attached[1])
    return unique_ids


def edited_record(message, path, old):
    """What an edit does to the record `old`: (record, whether Janet's input changed), or None.

    Telegram also reports edits of what the bot does not use (a link preview,
    a live location), so an edit counts only when the text or caption, or the
    files, differ from the record's. An album item's own caption is kept up to
    date even when the album reads the same; that alone does not count.
    """
    text = message.get('text', message.get('caption', ''))
    old_text = record_raw_text(old)
    old_images = old.get('images') or []
    old_ids = old.get('file_unique_ids')
    old_files = list(zip(old_images, old_ids or [None] * len(old_images)))
    edit_date = message.get('edit_date') or int(time.time())
    media_group_id = message.get('media_group_id')

    if media_group_id is not None:
        # An album keeps the images it was saved with; the edit carries only
        # its own item, and only that item's caption can change.
        captions = old.get('captions')
        if captions is not None:
            captions = dict(captions)
            captions[str(message['message_id'])] = text
            text = album_text(captions)
            if text == old_text:
                if captions == old['captions']:
                    return None
                return dict(old, captions=captions), False
        elif not text or text == old_text:
            # Saved before items kept their own captions: an item without one
            # says nothing about the album's.
            return None
        record = human_record(message, record_message_id(path, old), text, old_files,
                              edit_date=edit_date)
        record['media_group_id'] = media_group_id
        if captions is not None:
            record['captions'] = captions
        return record, True

    kept_reply = []
    if 'reply_to_message' not in message:
        replied = re.search(r'\nreply_to_message: (\d+)\n', old.get('text', ''))
        if replied:
            # Telegram leaves the reply out once the replied-to message is gone,
            # and an edit cannot change what a message replies to: keep it. A
            # message's own file comes first in the record.
            message = dict(message, reply_to_message={'message_id': int(replied.group(1))})
            kept_reply = old_files[1:] if panthera.attached_file(message) else old_files
    new_ids = attached_unique_ids(message) + [unique_id for _, unique_id in kept_reply]
    if old_ids is not None and new_ids == old_ids:
        # The same files: keep their paths rather than ask the Bot API again.
        files = old_files
        files_changed = False
    else:
        files = attached_files(message) + kept_reply
        files_changed = old_ids is not None or [p for p, _ in files] != old_images
    if not files_changed and text == old_text:
        return None
    return human_record(message, record_message_id(path, old), text, files,
                        edit_date=edit_date), True


@app.post("/edited_message")
async def call_edited_message(request: Request):
    """An edited message: rewrite its history record in place. It is never answered.

    A sent answer is final, and an edit neither summons Janet (not even one
    that adds /*) nor runs a command. An answer being generated from a history
    that contains the message is regenerated (edits.py).
    """
    empty = JSONResponse(content={
        "type": "empty",
        "body": ''
    })
    message = await request.json()
    chat_id = message['chat']['id']
    message_id = message['message_id']
    logger.info(f"call_edited_message: chat {chat_id}, message {message_id}, "
                f"edit_date {message.get('edit_date')}")

    if not user_access(message):
        return empty
    if not any(key in message for key in ('text', 'caption', 'photo', 'document')):
        return empty

    # From here on nothing awaits: the record and the running generations
    # learn about the edit together.
    media_group_id = message.get('media_group_id')
    buffered = media_group_buffers.get(media_group_id) if media_group_id is not None else None
    if buffered is not None:
        item = buffered['items'].get(message_id)
        if item is not None:
            item['caption'] = message.get('text', message.get('caption', ''))
            logger.info(f'edit of message {message_id}: album {media_group_id} is still '
                        f'being collected, caption updated')
        return empty

    # An album is filed under its first item, and an album's items have
    # consecutive ids, ten at most.
    neighbours = range(message_id - 1, message_id - 10, -1) if media_group_id is not None else ()
    found = panthera.find_human_records(chat_id, [message_id, *neighbours])
    matches = found.get(message_id, [])
    for first_item in neighbours:
        if matches:
            break
        matches = [(path, record) for path, record in found.get(first_item, [])
                   if record.get('media_group_id') == media_group_id]
    if not matches:
        logger.info(f'edit of message {message_id} in chat {chat_id}: not in the history '
                    f'(a command, pruned, or before a /reset)')
        return empty

    for path, old in matches:
        result = edited_record(message, path, old)
        if result is None:
            logger.info(f'edit of message {message_id} in chat {chat_id}: nothing Janet reads changed')
            continue
        record, changed = result
        panthera.rewrite_record(path, record)
        if not changed:
            logger.info(f'edit of message {message_id} in chat {chat_id}: album caption noted, '
                        f'the album reads the same')
            continue
        logger.info(f'edit of message {message_id} in chat {chat_id}: rewrote {os.path.basename(path)}')
        edits.record_edit(chat_id, record['message_id'])
    return empty


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

        # Iterate all possible chats that bot participates in
        chats_folder = 'data/users/'
        # Ensure folder exists; otherwise nothing to return
        if not os.path.isdir(chats_folder):
            logger.info(f"Chats folder not found: {chats_folder}")
            return JSONResponse(content={"status": "ok"})

        # Read list of entries in chats_folder
        entries = os.listdir(chats_folder)

        inline_elements = []
        # Iterate all entries and treat only valid numeric chat IDs as candidates
        for entry in entries:
            folder = entry.strip()
            # Only accept Telegram chat IDs like "-100123..." or "12345"
            if not re.fullmatch(r"-?\d+", folder):
                logger.debug(f"Skipping non-chat entry: {folder}")
                continue
            try:
                chat = bot.get_chat(int(folder))
            except Exception as e:
                logger.warning(f"Skipping entry {folder}: cannot get chat ({e})")
                continue

            if chat.type in ('group', 'supergroup'):
                logger.info(f"*** chat.title: {chat.title}")
                uid = hashlib.md5(folder.encode()).hexdigest()
                element = telebot.types.InlineQueryResultArticle(
                    id=uid,
                    title=chat.title,
                    input_message_content=telebot.types.InputTextMessageContent(f"response:{folder}"),
                )
                inline_elements.append(element)
            # else: skipped non-group chat types

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
