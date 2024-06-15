import os
import logging
import json
import requests
import time
import glob
import json
import logging
from pydantic import BaseModel, Field
from typing import List
# from langchain.agents import Tool, initialize_agent
from langchain.agents import Tool
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.tools import StructuredTool
# from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema import HumanMessage, AIMessage
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import YouTubeSearchTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
# from langchain.chains import RetrievalQA
from langchain_experimental.utilities import PythonREPL
import time as py_time
from pathlib import Path
import tiktoken
# from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts.chat import ChatPromptTemplate
import base64
from openai import OpenAI
import telebot
from telebot.formatting import escape_markdown
import logging

with open('config.json') as config_file:
    bot = telebot.TeleBot(json.load(config_file)['TOKEN'])

class TextOutput(BaseModel):
    text: str = Field(description="Text output")

class BotActionType(BaseModel):
    val: str = Field(description="Tool parameter value")

class image_context_conversation_args(BaseModel):
    text_request: str = Field(description="Text request in context of images")
    file_list: List[str] = Field(description="List of file_id")

class text_file_reader_args(BaseModel):
    file_list: List[str] = Field(description="List of file_id")

class ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    style: str = Field(description="The style of the generated images. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.")
    chat_id: str = Field(description="chat_id")
    message_id: str = Field(description="message_id")

def append_message(messages, role, text, image_url):
    messages.append(
        {
            "role": role,
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
            ],
        }
    )

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ChatAgent:
    def __init__(self, retriever, bot_instance):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.config = bot_instance.config
        self.retriever = retriever
        self.bot_instance = bot_instance  # Passing the Bot instance to the ChatAgent
        self.agent_executor = None
        # self.logger.info(f"ChatAgent function: {self.bot_instance.bot_action_come}")
        # self.agent = self.initialize_agent()
        self.initialize_agent()
        

    def initialize_agent(self):
        # model = 'gpt-4o-2024-05-13'
        model = 'gpt-4o'
        temperature = 0.7
        llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
            model=model,
            temperature=temperature,
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
            # return_direct=False,
        )
        # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        google_search = GoogleSerperAPIWrapper()
        
        google_search_tool = Tool(
            name="google_search",
            func=google_search.run,
            description="Useful to search in Google. Use by default. Provide links if possible.",
        )
        
        youtube = YouTubeSearchTool()
        youtube_tool = Tool(
            name="youtube_search",
            description="Useful for when the user explicitly asks you to look on Youtube. Provide links if possible.",
            func=youtube.run,
            # return_direct=False,
        )

        wolfram = WolframAlphaAPIWrapper()
        wolfram_tool = Tool(
                name="wolfram_alpha",
                func=wolfram.run,
                description="Useful when need to calculate the math expression or solve any scientific task. Provide the solution details if possible.",
            )
        
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wikipedia_tool = Tool(
                name="wikipedia_search",
                func=wikipedia.run,
                description="Useful when users request biographies or historical moments. Provide links if possible.",
            )

        # coroutine=self.image_context_conversation, # may be used instead of func
        image_context_conversation_tool = StructuredTool.from_function(
            # func=self.image_context_conversation,
            coroutine=self.image_context_conversation,
            name="image_context_conversation",
            description="Answering on your text request about provided images",
            args_schema=image_context_conversation_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        
        image_plotter_tool = StructuredTool.from_function(
            coroutine=self.ImagePlotterTool,
            name="image_plotter",
            description="A tool to generate and send to user images based on a given prompt",
            args_schema=ImagePlotterArgs,
            # return_direct=False,
        )

        text_file_reader_tool = StructuredTool.from_function(
            coroutine=self.text_file_reader,
            name="read_text_file",
            description="Provides the content of the text file",
            args_schema=text_file_reader_args,
        )

        tools = []
        tools.append(repl_tool)
        tools.append(wolfram_tool)
        tools.append(youtube_tool)
        tools.append(google_search_tool)
        tools.append(wikipedia_tool)
        tools.append(image_context_conversation_tool)
        tools.append(image_plotter_tool)
        tools.append(text_file_reader_tool)

        """tools.append(
            Tool(
                args_schema=DocumentInput,
                name='Knowledge base',
                description="Providing a game information from the knowledge base",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever),
            )
        )"""
        markdown_sample = """&&&bold text&&&
%%%italic text%%%
@@@underline@@@
~~~strikethrough~~~
||spoiler||
```
pre-formatted fixed-width code block
```"""
        system_prompt = f"""Your name is Janet.
You are Artificial Intelligence and the participant in the multi-user or personal telegram chat.
Your model is {model} with temperature: {temperature}.
You can determine the current date from the message_date field in the current message.
For the formatting you can use the telegram MarkdownV2 format. For example: {markdown_sample}."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        # self.agent_executor = AgentExecutor(
        #     agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, early_stopping_method="generate", max_iterations=20
        # )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    async def ImagePlotterTool(self, prompt: str, style: str, chat_id: str, message_id: str) -> str:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        style = style.lower()
        if style not in ["vivid", "natural"]:
            self.logger.info(f"Style {style} is not supported. Using default style: vivid")
            style = "vivid"

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            style=style,
            size="1024x1024",            
            quality="hd",
            user_id=chat_id,
            n=1,
        )
        self.logger.info(f"ImagePlotterTool response: {response}")
        image_url = response.data[0].url

        # Download the image
        image_data = requests.get(image_url).content

        caption = f"||{escape_markdown(response.data[0].revised_prompt)}||"
        self.logger.info(f"ImagePlotterTool caption: {caption}")

        # Send the photo
        bot.send_photo(
            chat_id=chat_id, 
            photo=image_data, 
            reply_to_message_id=message_id, 
            caption=caption,
            parse_mode="MarkdownV2"
            )
        
        return "Image generated and sent to the chat"
    
    async def image_context_conversation(self, text_request: str, file_list: List[str]):
        self.logger.info(f"image_context_conversation request: {text_request}; file_list: {file_list}")
        messages = []
        for file_path in file_list:
            self.logger.info(f"file_path: {file_path}")
            base64_image = encode_image(file_path)
            image_url = f"data:image/jpeg;base64,{base64_image}"    
            append_message(
                messages, 
                "user",
                text_request,
                image_url
            )
        api_key = os.environ.get('OPENAI_API_KEY', '')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        model = "gpt-4o"

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                # "max_tokens": 2000
            }
        )
        self.logger.info(f"image_context_conversation response text: {response.text}")
        response_text = json.loads(response.text)['choices'][0]['message']['content']
        return response_text
        # return "Это кот"
    
    async def text_file_reader(self, file_list: List[str]):
        self.logger.info(f"text_file_reader request: file_list: {file_list}")
        text = ""
        for file_path in file_list:
            self.logger.info(f"file_path: {file_path}")
            with open(file_path, 'r') as file:
                text += f"file_path: {file_path}\n{file.read()}"
        return text
        

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

    def is_reply_to_ai_message(self, message):
        if "reply_to_message" not in message:
            return False
        if "from" not in message["reply_to_message"]:
            return False
        if "is_bot" not in message["reply_to_message"]["from"]:
            return False
        if message["reply_to_message"]["from"]["is_bot"] == False:
            return False
        if "username" not in message["reply_to_message"]["from"]:
            return False
        if message["reply_to_message"]["from"]["username"] == os.environ.get('BOT_USERNAME', 'your_bot_name'):
            return True

        return False

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
        # Create folder if not exist
        Path(chat_path).mkdir(parents=True, exist_ok=True)
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
        list_of_files.sort(key=os.path.getctime, reverse=True)
        tokens = 0
        # Log list of files
        self.logger.info(f"list_of_files: \n{list_of_files}")
        # Iterate over sorted files and append message to messages list
        for file in list_of_files: 
            if tokens > self.config['token_limit']:
            # if tokens > 4000:
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
       
    def save_to_chat_history(
            self, 
            chat_id, 
            message_text, 
            message_id,
            type,
            message_date = None,
            name_of_user = 'AI'
            ):
        self.logger.info(f'save_to_chat_history: {chat_id} message_text: {message_text}')
        # Prepare a folder
        path = f'./data/chats/{chat_id}'
        os.makedirs(path, exist_ok=True)
        if message_date is None:
            message_date = py_time.strftime('%Y-%m-%d-%H-%M-%S', py_time.localtime())
        log_file_name = f'{message_date}_{message_id}.json'        

        chat_log_path = os.path.join(self.data_dir, str(chat_id))
        Path(chat_log_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(chat_log_path, log_file_name), 'w') as log_file:
            json.dump({
                "type": type,
                "text": f"{message_text}"
                }, log_file)
            
    def get_message_file_list(self, bot, message):
        if 'photo' in message or 'document' in message:
            file_id = ''            
            if 'photo' in message:
                photo = message['photo']
                self.logger.info(f"photo in message: {len(photo)}")
                if len(photo) > 0:
                    # Photo is a photo
                    file_id = photo[-1]['file_id']
                    self.logger.info("file_id: "+str(file_id))

            elif 'document' in message:
                self.logger.info("document in message")
                document = message['document']
                if document['mime_type'].startswith('image/'):
                    # Document is a photo
                    file_id = document['file_id']
                    self.logger.info("file_id: "+str(file_id))
                elif document['mime_type'].startswith('text/') or \
                    document['mime_type'].startswith('application/json') or \
                    document['mime_type'].startswith('application/xml'):
                    # Document is a text file
                    file_id = document['file_id']
                    self.logger.info("file_id: "+str(file_id))
            if file_id != '':
                file_info = bot.get_file(file_id)
                file_path = file_info.file_path
                self.logger.info(f'file_path: {file_path}')
                return f'[{file_path}]'
        return ''

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

    def get_first_name(self, message):
        # Define the first name of the user
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        elif 'username' in message['from']:
            first_name = message['from']['username']
        elif 'id' in message['from']:
            first_name = message['from']['id']
        else:
            first_name = 'Unknown'
        return first_name

    async def llm_request(self, bot, user_session, message, message_text, system_content=None):
        chat_id = message['chat']['id']
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        # response = await self.chat_agent.agent_executor.ainvoke(
        #     {
        #         "input": message_text,
        #         "chat_history": self.chat_history,
        #     }
        # )["output"]
        result = await self.chat_agent.agent_executor.ainvoke(
            {
                "input": message_text,
                "chat_history": self.chat_history,
            }
        )
        response = result["output"]
        
        self.save_to_chat_history(
            message['chat']['id'],
            response,
            message["message_id"],
            'AIMessage'
            )

        return response
