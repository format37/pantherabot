import os
import logging
import json
import requests
import time


class Panthera:
    
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def get_message_type(self, user_session, text):
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


    def save_user_session(self, user_id, session):
        self.logger.info(f'save_user_session: {user_id} with cmd: {session["last_cmd"]}')
        # Save the user json file
        path = './data/users'
        user_path = os.path.join(path, f'{user_id}.json')
        json.dump(session, open(user_path, 'w'))


    def get_user_session(self, user_id):
        self.logger.info(f'get_user_session: {user_id}')
        
        # Check is the usef json file exist
        path = './data/users'
        user_path = os.path.join(path, f'{user_id}.json')
        if not os.path.exists(user_path):
            default_path = os.path.join(path, 'default.json')
            session = json.load(open(default_path, 'r'))
            # Save the user json file
            self.save_user_session(user_id, session)

        session = json.load(open(user_path, 'r'))
        # Return the user json file as dict
        return session


    def log_message(self, message):
        self.logger.info(f'message: {message}')
        # Read the chat id from the message
        chat_id = message['chat']['id']
        # Prepare a folder
        path = f'./data/chats/{chat_id}'
        os.makedirs(path, exist_ok=True)
        filename = f'{message["date"]}_{message["message_id"]}.json'
        # Save the user json file
        file_path = os.path.join(path, filename)
        json.dump(message, open(file_path, 'w'))


    def reset_chat(self, chat_id):
        self.logger.info(f'reset_chat: {chat_id}')
        chat_path = f'./data/chats/{chat_id}'
        # Remove all files in chat path
        for f in os.listdir(chat_path):
            self.logger.info(f'remove file: {f}')
            os.remove(os.path.join(chat_path, f))


    def token_counter(self, text, model):
        llm_url = os.environ.get('LLM_URL', '')
        url = f'{llm_url}/token_counter'
        data = {
            "text": text,
            "model": model
        }

        response = requests.post(url, json=data)
        # response = requests.post(url, kwargs=data)
        return response
    
    def default_bot_message(self, message, text):
        current_unix_timestamp = int(time.time())
        return {
        'message_id': int(message['message_id']) + 1,
        'from': {
                'id': 0, 
                'is_bot': True, 
                'first_name': 'assistant', 
                'username': 'assistant', 
                'language_code': 'en', 
                'is_premium': False
            }, 
            'chat': {
                'id': message['chat']['id'], 
                'first_name': message['chat']['first_name'], 
                'username': message['chat']['username'], 
                'type': 'private'
            }, 
            'date': current_unix_timestamp, 
            'text': text
        }
    

    def add_evaluation_to_topic(self, session, topic_name, value=10):
        """
        Function to add an evaluation to a specified topic in a session dictionary.
        
        Args:
        - session (dict): The session dictionary to modify.
        - topic_name (str): The name of the topic to add or modify.
        - date (str): The date for the evaluation. If None, use the current date.
        - value (int): The integer value for the evaluation.
        
        Returns:
        - dict: The modified session dictionary.
        """
        # Ensure "topics" is a dictionary
        if "topics" not in session:
            session["topics"] = {}
        
        # If the topic doesn't exist, add it
        if topic_name not in session["topics"]:
            session["topics"][topic_name] = {"evaluations": []}
        
        # Unix timestamp
        date = int(time.time())
        # Create evaluation dictionary
        evaluation_dict = {"date": date, "value": value}
        
        # Add evaluation to topic
        session["topics"][topic_name]["evaluations"].append(evaluation_dict)
        
        return session


    def llm_request(self, user_session, message, system_content=None):
        chat_id = message['chat']['id']
        self.logger.info(f'llm_request: {chat_id}')
        # Prepare a folder
        path = f'./data/chats/{chat_id}'
        # Read files in path, sorted by name ascending
        files = sorted(os.listdir(path), reverse=False)
        
        # Fill the prompt
        if system_content is None:
            system_content = "You are the chat member. Your username is assistant. You need to start with 'Assistant:' before each of your messages."
        prompt = [
            {"role": "system", "content": system_content}
        ]

        for file in files:
            # Extract the text from the json file
            message = json.load(open(os.path.join(path, file), 'r'))
            """
            {
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
            # Extract the text from the message
            text = message['text']
            if message['from']['id']==0:
                role = 'assistant'                
            else:
                role = 'user'
                user_name = message['from']['first_name']
                if message['from']['first_name'] == '':
                    user_name = message['from']['username']
                    if message['from']['username'] == '':
                        user_name = 'Unknown'
                # Add preamble to the message
                preamble = f'{user_name}: '
                text = preamble + message['text']

            prompt.append({"role": role, "content": text})

        # Read the last file
        # last_file = files[-1]
        
        # Extract the text from the last json file
        # message = json.load(open(os.path.join(path, last_file), 'r'))
        # Extract the text from the message
        # user_text = message['text']
        llm_url = os.environ.get('LLM_URL', '')
        url = f'{llm_url}/request'
        """prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text}
        ]"""
        request_data = {
            "api_key": os.environ.get('LLM_TOKEN', ''),
            "model": user_session['model'],
            "prompt": prompt
        }
        # Json dumps prompt
        prompt_dumped = json.dumps(prompt)
        tokens_count = self.token_counter(prompt_dumped, user_session['model']).json()['tokens']
        self.logger.info(f'tokens_count prognose: {tokens_count}')
        self.logger.info(f'request_data: {request_data}')
        response = requests.post(url, json=request_data)
        self.logger.info(f'response: {str(response)}')

        # Get the current time in Unix timestamp format

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

        bot_message = self.default_bot_message(
            message,
            response_json['choices'][0]['message']['content']
            )
        # Log message
        self.log_message(bot_message)
        # Remove left 11 signs: 'assistant: '
        if response_json['choices'][0]['message']['content'].startswith('assistant: '):
            response_text = response_json['choices'][0]['message']['content'][11:]
        else:
            response_text = response_json['choices'][0]['message']['content']

        # Return the response
        return response_text