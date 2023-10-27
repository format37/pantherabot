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


    def save_user_session(self, user_id, session):
        self.logger.info(f'save_user_session: {user_id}')
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
            save_user_session(user_id, session)

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


    def llm_request(self, user_session, chat_id, text):
        self.logger.info(f'llm_request: {chat_id}')
        # Prepare a folder
        path = f'./data/chats/{chat_id}'
        # Read files in path, sorted by name ascending
        files = sorted(os.listdir(path), reverse=False)
        
        # Fill the prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. You need to add 'assistant: ' to the beginning of your message."}
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
            user_text = message['text']
            if message['from']['id']==0:
                role = 'assistant'
            else:
                role = 'user'

            prompt.append({"role": role, "content": user_text})

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
        current_unix_timestamp = int(time.time())

        response_json = json.loads(response.text)

        # Log message
        bot_message = {
        'message_id': int(message['message_id']) + 1,
        'from': {
                'id': 0, 
                'is_bot': True, 
                'first_name': 'assistant', 
                'username': 'assistant', 
                'language_code': 'en', 
                'is_premium': True
            }, 
            'chat': {
                'id': 106129214, 
                'first_name': 'Alex', 
                'username': 'format37', 
                'type': 'private'
            }, 
            'date': current_unix_timestamp, 
            'text': response_json['choices'][0]['message']['content']
        }
        # Log message
        self.log_message(bot_message)

        # Return the response
        return response