import os
import logging
import json


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_user_session(user_id, session):
    logger.info(f'save_user_session: {user_id}')
    # Save the user json file
    path = './data/users'
    user_path = os.path.join(path, f'{user_id}.json')
    json.dump(session, open(user_path, 'w'))


def get_user_session(user_id):
    logger.info(f'get_user_session: {user_id}')
    
    # Check is the usef json file exist
    path = './data/users'
    user_path = os.path.join(path, f'{user_id}.json')    
    if not os.path.exists(user_path):
        default_path = os.path.join(path, 'default.json')
        session = json.load(open(default_path, 'r'))
        # Save the user json file
        save_user_session(user_id, session)

    session = json.load(open(user_path, 'r'))
    # Return the user json file as dict
    return session


def log_message(message):
    logger.info(f'message: {message}')
    # Read the chat id from the message
    chat_id = message['chat']['id']
    # Prepare a folder
    path = f'./data/chats/{chat_id}'
    os.makedirs(path, exist_ok=True)
    filename = f'{message["date"]}_{message["message_id"]}.json'
    # Save the user json file
    file_path = os.path.join(path, filename)
    json.dump(message, open(file_path, 'w'))


def reset_chat(chat_id):
    logger.info(f'reset_chat: {chat_id}')
    chat_path = f'./data/chats/{chat_id}'
    # Remove all files in chat path
    for f in os.listdir(chat_path):
        logger.info(f'remove file: {f}')
        os.remove(os.path.join(chat_path, f))


def llm_request(user_session, chat_id, text):
    logger.info(f'llm_request: {chat_id}')
    # Prepare a folder
    path = f'./data/chats/{chat_id}'
    # Read files in path, sorted by name ascending
    files = sorted(os.listdir(path), reverse=False)
    # Read the last file
    last_file = files[-1]
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
    # Extract the text from the last json file
    message = json.load(open(os.path.join(path, last_file), 'r'))
    # Extract the text from the message
    user_text = message['text']

    url = os.environ.get('LLM_URL', '')
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]
    request_data = {
        "api_key": os.environ.get('LLM_TOKEN', ''),
        "model": user_session['model'],
        "prompt": prompt
    }
    # Json dumps prompt
    prompt_dumped = json.dumps(prompt)
    print(
        'Token count forecast:', 
        token_counter(prompt_dumped, model).json()['tokens']
        )
    response = requests.post(url, json=request_data)
    logger.info(f'response: {str(response)}')
    """
    # Prepare the request
    request =  [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user", 
            "content": text
        }
    ]
    # Send the request
    logger.info(f'request: {request}')
    response = requests.post('http://localhost:5000/llm', json=request)
    """

    # Return the response
    return response