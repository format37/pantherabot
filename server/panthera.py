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
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import YouTubeSearchTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_experimental.utilities import PythonREPL
import time as py_time
from pathlib import Path
import tiktoken
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts.chat import ChatPromptTemplate
import base64
from openai import OpenAI
import telebot
from telebot.formatting import escape_markdown
import logging
import re

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

class update_system_prompt_args(BaseModel):
    chat_id: str = Field(description="chat_id")
    new_prompt: str = Field(description="new_prompt")

class reset_system_prompt_args(BaseModel):
    chat_id: str = Field(description="chat_id")

class ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    style: str = Field(description="The style of the generated images. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.")
    chat_id: str = Field(description="chat_id")
    message_id: str = Field(description="message_id")

class BFL_ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    chat_id: int = Field(description="chat_id")
    message_id: int = Field(description="message_id")
    raw_mode: bool = Field(description="raw_mode Generate less processed, more natural-looking images")

class ask_reasoning_args(BaseModel):
    request: str = Field(description="Request for the Reasoning expert")

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
        # model="claude-3-5-sonnet-20240620",  # Specify the model name you want to use
        # llm = ChatAnthropic(
        #     model="claude-3-5-sonnet-20241022",
        #     temperature = 0.7,
        #     max_tokens=4096,
        # )
        # if "ANTHROPIC_API_KEY" not in os.environ:
        #     self.logger.error("ANTHROPIC_API_KEY is not set")

        # temperature=0.7,
        #     max_tokens=150,
        #     max_retries=2,
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
            coroutine=self.BFL_ImagePlotterTool,
            name="image_plotter",
            description="A tool to generate and send to user images based on a given prompt",
            args_schema=BFL_ImagePlotterArgs,
            # return_direct=False,
        )

        text_file_reader_tool = StructuredTool.from_function(
            coroutine=self.text_file_reader,
            name="read_text_file",
            description="Provides the content of the text file",
            args_schema=text_file_reader_args,
        )

        update_system_prompt_tool = StructuredTool.from_function(
            coroutine=self.update_system_prompt,
            name="update_system_prompt",
            description="Update the system prompt for the corresponding group_id",
            args_schema=update_system_prompt_args
        )

        reset_system_prompt_tool = StructuredTool.from_function(
            coroutine=self.reset_system_prompt,
            name="reset_system_prompt",
            description="Reset the system prompt for the corresponding group_id",
            args_schema=reset_system_prompt_args
        )

        ask_reasoning_tool = StructuredTool.from_function(
            coroutine=self.ask_reasoning,
            name="ask_reasoning",
            description="Ask the Reasoning expert LLM for the given request",
            args_schema=ask_reasoning_args
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
        tools.append(update_system_prompt_tool)
        tools.append(reset_system_prompt_tool)
        tools.append(ask_reasoning_tool)

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
                # ("system", system_prompt),
                ("system", "{system_prompt}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def bfl_generate_image(self, headers, prompt, width=1024, height=768):
        """
        Submit an image generation request to the API.
        """
        API_URL = "https://api.bfl.ml"
        # endpoint = f"{API_URL}/v1/flux-pro-1.1"
        endpoint = f"{API_URL}/v1/flux-pro-1.1-ultra"
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "safety_tolerance": 6,
            "raw": True,
            "output_format": "png",
        }

        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["id"]

    def bfl_get_result(self, task_id, headers):
        """
        Retrieve the result of an image generation task.
        """
        API_URL = "https://api.bfl.ml"
        endpoint = f"{API_URL}/v1/get_result"
        params = {"id": task_id}

        while True:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()

            if result["status"] == "Ready":
                return result["result"]
            elif result["status"] in ["Error", "Request Moderated", "Content Moderated"]:
                raise Exception(f"Task failed with status: {result['status']}")
            
            time.sleep(5)  # Wait for 5 seconds before checking again

    async def BFL_ImagePlotterTool(self, prompt: str, chat_id: int, message_id: int, raw_mode: bool) -> str:        
        headers = {
            "x-key": os.environ.get('BFL_API_KEY', ''),
            "Content-Type": "application/json"
        }
        self.logger.info(f"Submitting the bfl image generation request...")
        task_id = self.bfl_generate_image(headers, prompt, 1280, 1280, raw_mode)
        self.logger.info(f"Task ID: {task_id} Waiting for the image to be generated...")
        result = self.bfl_get_result(task_id, headers)
        if "sample" in result:
            image_url = result["sample"]
            # Download the image
            image_data = requests.get(image_url).content
            
            # Save the image to the user's images directory
            image_dir = f"data/users/{chat_id}/images"
            os.makedirs(image_dir, exist_ok=True)
            # image_path = os.path.join(image_dir, f"{int(time.time())}.jpg")
            # with open(image_path, 'wb') as f:
            #     f.write(image_data)

            caption = f"||{escape_markdown(prompt)}||"
            self.logger.info(f"ImagePlotterTool caption: {caption}")

            # Send the photo
            sent_message = bot.send_photo(
                chat_id=chat_id, 
                photo=image_data, 
                reply_to_message_id=message_id, 
                caption=caption,
                parse_mode="MarkdownV2"
                )
            file_id = sent_message.photo[-1].file_id
            self.logger.info(f"sent_message file_id: {file_id}")
            # file_info = bot.get_file(file_id)
            # file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
            # self.logger.info(f"file_url: {file_url}")

            image_path = os.path.join(image_dir, f"{file_id}")
            # write empty file
            with open(image_path, 'w') as f:
                f.write("")
            
            return "Image generated and sent to the chat"
        else:
            return f"Image generation failed: {result}"
    
    async def ImagePlotterTool_x(self, prompt: str, style: str, chat_id: str, message_id: str) -> str:
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
            user=chat_id,
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
    

    async def update_system_prompt(self, chat_id, new_prompt):
        self.logger.info(f"[chat_id] update_system_prompt: {new_prompt}")
        # Save the new system prompt
        os.makedirs('./data/custom_prompts', exist_ok=True)
        with open(f'./data/custom_prompts/{chat_id}.txt', 'w') as f:
            f.write(new_prompt)
        return "System prompt updated"
        
    async def reset_system_prompt(self, chat_id):
        self.logger.info(f"[chat_id] reset_system_prompt")
        # Remove the custom system prompt
        custom_prompt_path = f'./data/custom_prompts/{chat_id}.txt'
        if os.path.exists(custom_prompt_path):
            os.remove(custom_prompt_path)
        return "System prompt reset: ok"
    
    async def ask_reasoning(self, request):
        self.logger.info(f"[ask_reasoning] request")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
        model="o1-preview",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"{request}"
                        },                
                    ]
                },
            ],
        )
        return response.choices[0].message.content

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
        message_date=None,
        name_of_user='AI'
    ):
        user_id = chat_id  # Assuming chat_id corresponds to user_id in private chats
        chat_log_path = os.path.join('data', 'users', str(user_id), 'chats', str(chat_id))
        os.makedirs(chat_log_path, exist_ok=True)
        if message_date is None:
            message_date = py_time.strftime('%Y-%m-%d-%H-%M-%S', py_time.localtime())
        log_file_name = f'{message_date}_{message_id}.json'
        with open(os.path.join(chat_log_path, log_file_name), 'w') as log_file:
            json.dump({
                "type": type,
                "text": f"{message_text}"
            }, log_file)

    def save_to_chat_history_x(
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
        user_id = chat_id  # Assuming chat_id corresponds to user_id in private chats
        chat_log_path = os.path.join('data', 'users', str(user_id), 'chats', str(chat_id))
        if not os.path.exists(chat_log_path):
            return
        self.chat_history = []
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
                    # Remove corrupted file
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
    
    def get_system_prompt(self, chat_id):
        # Check if there's a custom system prompt for this chat
        custom_prompt_path = f'./data/custom_prompts/{chat_id}.txt'
        if os.path.exists(custom_prompt_path):
            with open(custom_prompt_path, 'r') as f:
                return f.read().strip()
        # If no custom prompt, return the default system prompt
        # return self.config['default_system_prompt']
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
You can determine the current date from the message_date field in the current message.
For the formatting you can use the telegram MarkdownV2 format. For example: {markdown_sample}."""
        return system_prompt

    # async def llm_request(self, bot, message, message_text):
    async def llm_request(self, chat_id, message_id, message_text):
        # message_text may have augmentations
        # chat_id = message['chat']['id']
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id)
        result = await self.chat_agent.agent_executor.ainvoke(
            {
                "input": message_text,
                "chat_history": self.chat_history,
                "system_prompt": system_prompt,
            }
        )
        response = result["output"]
        
        
        # if response is a list
        if isinstance(response, list):
            response = response[0]
            # if resonse is a dict
            if isinstance(response, dict):
                try:
                    response = response['text']
                except:
                    self.logger.info(f'llm_request response has no "text": {response}')
                    response = str(response)

        self.logger.info(f'llm_request response type: {type(response)}')
        self.logger.info(f'llm_request response: {response}')
        
        self.save_to_chat_history(
            # message['chat']['id'],
            chat_id,
            response,
            # message["message_id"],
            message_id,
            'AIMessage'
            )

        return response

    class Filename(BaseModel):
        name: str = Field(..., description="The generated filename without extension")

    async def generate_filename(self, content):
        # Truncate content if it's too long
        max_content_length = 1000
        truncated_content = content[:max_content_length] + "..." if len(content) > max_content_length else content

        # client = OpenAI()
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise and relevant file names based on given content."},
                    {"role": "user", "content": f"Generate a short, descriptive filename (without extension) for a text file containing the following content:\n\n{truncated_content}\n\nThe filename should be concise, relevant, and use underscores instead of spaces."}
                ],
                response_format=self.Filename,
            )

            message = completion.choices[0].message
            if message.parsed:
                filename = message.parsed.name

                # Clean the filename
                filename = re.sub(r'[^\w\-_\.]', '_', filename)  # Replace invalid characters with underscore
                filename = re.sub(r'_+', '_', filename)  # Replace multiple underscores with single underscore
                filename = filename.strip('_')  # Remove leading/trailing underscores

                return filename + ".txt"
            else:
                print(f"Error generating filename: {message.refusal}")
                return "response.txt"  # Fallback to default name if there's an error
        except Exception as e:
            print(f"Error generating filename: {e}")
            return "response.txt"  # Fallback to default name if there's an error