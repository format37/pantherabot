import os
import logging
import json
import requests
import time
import glob
import json
import logging
from pydantic import BaseModel, Field
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.tools import StructuredTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import RetrievalQA
from langchain.tools import Tool
# from langchain.schema import TextOutput
from langchain_experimental.utilities import PythonREPL
import time as py_time
from pathlib import Path
import tiktoken
# import webbrowser as wb

class TextOutput(BaseModel):
    text: str = Field(description="Text output")

class BotActionType(BaseModel):
    val: str = Field(description="Tool parameter value")

class ChatAgent:
    def __init__(self, retriever, bot_instance):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.config = bot_instance.config
        self.retriever = retriever
        self.bot_instance = bot_instance  # Passing the Bot instance to the ChatAgent
        # self.logger.info(f"ChatAgent function: {self.bot_instance.bot_action_come}")
        self.agent = self.initialize_agent()
        

    def initialize_agent(self):
        llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
            model=self.config['model'],
            temperature=self.config['temperature'],
        )
        # llm = Ollama(model="llama2")
        # llm = Ollama(model="mistral")
        tools = []
        python_repl = PythonREPL()
        # Non direct return
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
            return_direct=False,
        )
        tools.append(repl_tool)
        """tools = [self.create_structured_tool(func, name, description, return_direct)
                 for func, name, description, return_direct in [
                        (self.bot_instance.web_browser_tool, "Web browsing",
                            "Provide a link to request", True)
                      ]
                 ]"""
        # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        # web_browsing_tool = SimulatedWebBrowsingTool(llm, embeddings)
        # tools.append(web_browsing_tool)
        # tools.append(DuckDuckGoSearchRun())
        tools.append(DuckDuckGoSearchResults())
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(wikipedia)
        """tools.append(
            Tool(
                args_schema=DocumentInput,
                name='Knowledge base',
                description="Providing a game information from the knowledge base",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever),
            )
        )"""
        return initialize_agent(
            tools,
            llm,
            agent='chat-conversational-react-description',
            verbose=True,
            handle_parsing_errors=True
        )

    @staticmethod
    def create_structured_tool(func, name, description, return_direct):
        print(f"create_structured_tool name: {name} func: {func}")
        return StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=BotActionType,
            return_direct=return_direct,
        )

class Panthera:
    
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.config = json.load(open('./data/users/default.json', 'r'))
        self.chat_agent = ChatAgent(None, self)
        self.data_dir = './data/chats'
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)  # Ensure data directory exists
        self.chat_history = []

    """def web_browser_tool(self, bot_action_type: BotActionType):
        self.logger.info(f"web_browser_tool: {bot_action_type}")
        result = wb.open(bot_action_type.val)
        return TextOutput(text=f"Web browser opened: {result}")"""

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

    def reset_chat(self, chat_id):
        self.logger.info(f'reset_chat: {chat_id}')
        chat_path = f'./data/chats/{chat_id}'
        # Remove all files in chat path
        for f in os.listdir(chat_path):
            self.logger.info(f'remove file: {f}')
            os.remove(os.path.join(chat_path, f))

    def token_counter(self, text):
        enc = tiktoken.encoding_for_model(self.config['model'])
        tokens = enc.encode(text)
        return len(tokens)
    
    def default_bot_message(self, message, text):
        current_unix_timestamp = int(time.time())
        self.logger.info(f'default_bot_message: {message}')
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        else:
            first_name = message['from']['username']
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
                'first_name': first_name, 
                'username': message['from']['username'], 
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
    
    def crop_queue(self, chat_id):
        """
        Function to remove the oldest messages from the chat queue until the token limit is reached.
        
        Args:
        - chat_id (str): The chat ID.
        - token_limit (int): The maximum number of tokens allowed in the queue.
        """
        # Create chat path
        chat_path = os.path.join("data", "chats", str(chat_id))
        # Create folder if not exist
        Path(chat_path).mkdir(parents=True, exist_ok=True)
        # Get all files in folder
        list_of_files = glob.glob(chat_path + "/*.json")
        # Sort files by creation time ascending
        # list_of_files.sort(key=os.path.getctime)
        # Sort files by creation time descending
        list_of_files.sort(key=os.path.getctime, reverse=True)
        tokens = 0
        # Log list of files
        self.logger.info(f"list_of_files: \n{list_of_files}")
        # Iterate over sorted files and append message to messages list
        for file in list_of_files: 
            if tokens > self.config['token_limit']:
                self.logger.info(f"Removing file: {file}")
                os.remove(file)
                continue
            # Load file
            try:
                message = json.load(open(file, 'r'))
                # Extract the text from the message
                text = message['text']
                # Get the number of tokens for the message
                tokens += self.token_counter(text)
                self.logger.info(f"file: {file} tokens: {tokens}")
                # If the token limit is reached, remove the file
                if tokens > self.config['token_limit']:
                    self.logger.info(f"Removing file: {file}")
                    os.remove(file)
            except Exception as e:
                self.logger.error(f"Error loading file: {file} error: {e}")
                os.remove(file)
    
    '''def read_latest_messages(self, user_session, message, system_content=None):
        # model = user_session['model']
        model = self.config['model']
        chat_id = message['chat']['id']
        user_id = message['from']['id']
        token_limit = 3000
        chat_gpt_prompt = []
        # Fill the prompt
        if system_content is None:
            system_content = "You are the chat member. Your username is assistant. You need to start with 'Assistant:' before each of your messages."
        chat_gpt_prompt_original = [
            {"role": "system", "content": system_content}
        ]
        # if chat_type == 'group' or chat_type == 'supergroup':
        if message['chat']['type'] != 'private':
            self.logger.info("read group chat")
            # Create group id folder in the data path if not exist
            path = os.path.join("data", "chats", str(chat_id))
            # Get all files in folder
            list_of_files = glob.glob(path + "/*.json")
        else:
            self.logger.info("read private chat")
            # Create user id folder in the data path if not exist
            path = os.path.join("data", "chats", str(user_id))
            # Get all files in folder
            list_of_files = glob.glob(path + "/*.json")

        # Sort files by creation time ascending
        list_of_files.sort(key=os.path.getctime, reverse=True)

        # Iterate over sorted files and append message to messages list
        limit_reached = False
        for file_name in list_of_files:
            self.logger.info("reading file: "+file_name)
            prompt_dumped = json.dumps(chat_gpt_prompt)
            if limit_reached == False and \
                self.token_counter(prompt_dumped).json()['tokens']<token_limit:
                
                with open(file_name, "r") as f:
                               
                    # Extract the text from the json file
                    # message = json.load(open(os.path.join(path, file), 'r'))
                    message = json.load(f)
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

                    chat_gpt_prompt.append({"role": role, "content": text})
            else:
                limit_reached = True
                self.logger.info("token limit reached. removing file: "+file_name)
                os.remove(file_name)

        # Sort chat_gpt_prompt reversed
        chat_gpt_prompt.reverse()
        # Now add all values of chat_gpt_prompt to chat_gpt_prompt_original
        for item in chat_gpt_prompt:
            chat_gpt_prompt_original.append(item)

        # logger.info("chat_gpt_prompt_original: "+str(chat_gpt_prompt_original))

        return chat_gpt_prompt_original'''
    
    # def log_message(self, chat_id: str, message_text: str):
    def save_to_chat_history(
            self, 
            chat_id, 
            message_text, 
            message_id,
            type,
            message_date = None,
            name_of_user = 'AI'
            ):
        # chat_id = message['chat']['id']
        # message_text = message['text']
        # Prepare a folder
        path = f'./data/chats/{chat_id}'
        os.makedirs(path, exist_ok=True)
        # filename = f'{message["date"]}_{message["message_id"]}.json'
        if message_date is None:
            message_date = py_time.strftime('%Y-%m-%d-%H-%M-%S', py_time.localtime())
        log_file_name = f'{message_date}_{message_id}.json'        

        chat_log_path = os.path.join(self.data_dir, str(chat_id))
        Path(chat_log_path).mkdir(parents=True, exist_ok=True)
        # timestamp = int(time.time())
        # log_file_name = f"{timestamp}.json"
        with open(os.path.join(chat_log_path, log_file_name), 'w') as log_file:
            json.dump({
                "type": type,
                "text": f"{message_text}"
                }, log_file)

    def read_chat_history(self, chat_id: str):
        '''Reads the chat history from a folder.'''
        self.chat_history = []
        chat_log_path = os.path.join(self.data_dir, str(chat_id))
        # Create the chat log path if not exist
        Path(chat_log_path).mkdir(parents=True, exist_ok=True)
        self.crop_queue(chat_id=chat_id)
        for log_file in sorted(os.listdir(chat_log_path)):
            with open(os.path.join(chat_log_path, log_file), 'r') as file:
                try:
                    message = json.load(file)
                    if message['type'] == 'AIMessage':
                        self.chat_history.append(AIMessage(content=message['text']))
                    elif message['type'] == 'HumanMessage':
                        self.chat_history.append(HumanMessage(content=message['text']))
                except Exception as e:
                    self.logger.error(f'Error reading chat history: {e}')
                    # Remove
                    os.remove(os.path.join(chat_log_path, log_file))

    # The original llm_request function now refactored with Langchain's conversational agent
    # def llm_request(chat_id: str, message_text: str, user_session) -> str:
    def llm_request(self, user_session, message, system_content=None):
        chat_id = message['chat']['id']
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)

        message_text = message['text']
        
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        elif 'username' in message['from']:
            first_name = message['from']['username']
        elif 'id' in message['from']:
            first_name = message['from']['id']
        else:
            first_name = 'Unknown'

        self.save_to_chat_history(
            message['chat']['id'], 
            message['text'], 
            message["message_id"],
            'HumanMessage',
            message["date"],
            first_name
            )

        # self.logger.info(f'sending:\n{message_text}')
        
        response = self.chat_agent.agent.run(
            input=message_text, 
            chat_history=self.chat_history
            )
        
        # self.logger.info(f'response:\n{response}')

        self.save_to_chat_history(
            message['chat']['id'],
            response,
            message["message_id"],
            'AIMessage'
            )

        return response
