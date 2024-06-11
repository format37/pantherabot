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
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_community.tools import StructuredTool
from langchain.tools.base import StructuredTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import YouTubeSearchTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.chains import RetrievalQA
from langchain_experimental.utilities import PythonREPL
import time as py_time
from pathlib import Path
import tiktoken
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.prompts.chat import ChatPromptTemplate
import base64
from openai import OpenAI
import telebot

with open('config.json') as config_file:
    bot = telebot.TeleBot(json.load(config_file)['TOKEN'])

class TextOutput(BaseModel):
    text: str = Field(description="Text output")

class BotActionType(BaseModel):
    val: str = Field(description="Tool parameter value")

class image_context_conversation_args(BaseModel):
    text_request: str = Field(description="Text request in context of images")
    file_list: List[str] = Field(description="List of file_id")

class ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    chat_id: str = Field(description="chat_id")
    message_id: str = Field(description="message_id")

markdown_sample = """*bold \*text*
_italic \*text_
__underline__
~strikethrough~
||spoiler||
*bold _italic bold ~italic bold strikethrough ||italic bold strikethrough spoiler||~ __underline italic bold___ bold*
[inline URL](http://www.example.com/)
[inline mention of a user](tg://user?id=123456789)
![👍](tg://emoji?id=5368324170671202286)
`inline fixed-width code`
```
pre-formatted fixed-width code block
```
```python
pre-formatted fixed-width code block written in the Python programming language
```
>Block quotation started
>Block quotation continued
>Block quotation continued
>Block quotation continued
>The last line of the block quotation
**>The expandable block quotation started right after the previous block quotation
>It is separated from the previous block quotation by an empty bold entity
>Expandable block quotation continued
>Hidden by default part of the expandable block quotation started
>Expandable block quotation continued
>The last line of the expandable block quotation with the expandability mark||"""

html_instruction = """<b>bold</b>, <strong>bold</strong>
<i>italic</i>, <em>italic</em>
<u>underline</u>, <ins>underline</ins>
<s>strikethrough</s>, <strike>strikethrough</strike>, <del>strikethrough</del>
<span class="tg-spoiler">spoiler</span>, <tg-spoiler>spoiler</tg-spoiler>
<b>bold <i>italic bold <s>italic bold strikethrough <span class="tg-spoiler">italic bold strikethrough spoiler</span></s> <u>underline italic bold</u></i> bold</b>
<a href="http://www.example.com/">inline URL</a>
<a href="tg://user?id=123456789">inline mention of a user</a>
<tg-emoji emoji-id="5368324170671202286">👍</tg-emoji>
<code>inline fixed-width code</code>
<pre>pre-formatted fixed-width code block</pre>
<pre><code class="language-python">pre-formatted fixed-width code block written in the Python programming language</code></pre>
<blockquote>Block quotation started\nBlock quotation continued\nThe last line of the block quotation</blockquote>
<blockquote expandable>Expandable block quotation started\nExpandable block quotation continued\nExpandable block quotation continued\nHidden by default part of the block quotation started\nExpandable block quotation continued\nThe last line of the block quotation</blockquote>

Please note:

    Only the tags mentioned above are currently supported.
    All <, > and & symbols that are not a part of a tag or an HTML entity must be replaced with the corresponding HTML entities (< with &lt;, > with &gt; and & with &amp;).
    All numerical HTML entities are supported.
    The API currently supports only the following named HTML entities: &lt;, &gt;, &amp; and &quot;.
    Use nested pre and code tags, to define programming language for pre entity.
    Programming language can't be specified for standalone code tags.
    A valid emoji must be used as the content of the tg-emoji tag. The emoji will be shown instead of the custom emoji in places where a custom emoji cannot be displayed (e.g., system notifications) or if the message is forwarded by a non-premium user. It is recommended to use the emoji from the emoji field of the custom emoji sticker.
    Custom emoji entities can only be used by bots that purchased additional usernames on Fragment.
"""
supported_html_tags = '<b><strong><i><em><u><ins><s><strike><del><span class="tg-spoiler"><tg-spoiler><b><a href="http://www.example.com/"><code><pre><code class="language-python">'

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
        llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
            model="gpt-4o-2024-05-13",
            temperature=0.7,
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
            func=self.image_context_conversation,
            name="image_context_conversation",
            description="Answering on your text request about provided images",
            args_schema=image_context_conversation_args,
            # return_direct=False,
        )
        
        image_plotter_tool = StructuredTool.from_function(
            func=self.ImagePlotterTool,
            name="image_plotter",
            description="A tool to generate and send to user images based on a given prompt",
            args_schema=ImagePlotterArgs,
            # return_direct=False,
        )

        tools = []
        tools.append(repl_tool)
        tools.append(wolfram_tool)
        tools.append(youtube_tool)
        tools.append(google_search_tool)
        tools.append(wikipedia_tool)
        tools.append(image_context_conversation_tool)
        tools.append(image_plotter_tool)

        """tools.append(
            Tool(
                args_schema=DocumentInput,
                name='Knowledge base',
                description="Providing a game information from the knowledge base",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever),
            )
        )"""
        prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", f"You are telegram chat member. Your may represent your answer in HTML format following this instruction:\n{html_instruction}."),
                ("system", "You are telegram chat member."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def ImagePlotterTool(self, prompt: str, chat_id: str, message_id: str) -> str:
        # name = "image_plotter"
        # description = "A tool to generate and save images based on a given prompt"
        # args_schema = ImagePlotterArgs

        # def _run(self, prompt: str, chat_id: str, message_id: str) -> str:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        self.logger.info(f"ImagePlotterTool response: {response}")
        image_url = response.data[0].url

        # Download the image
        image_data = requests.get(image_url).content

        # Send the photo
        bot.send_photo(chat_id=chat_id, photo=image_data, reply_to_message_id=message_id, caption=response.data[0].revised_prompt)
        
        return "Image generated and sent to the chat"

    def image_context_conversation(self, text_request: str, file_list: List[str]):
        # postfix = f". Your should represent your answer only in HTML format following this instruction:\n{html_instruction}."
        postfix = ""
        text_request = text_request + postfix
        self.logger.info(f"image_context_conversation request: {text_request}; file_list: {file_list}")
        messages = []
        for file_path in file_list:
            self.logger.info(f"file_path: {file_path}")
            # file_path = file_list[0]
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

        # model = "gpt-4o-2024-05-13"
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
        # response_content = response.choices[0].message.content
        # try:
        # response_text = response.text['choices'][0]['message']['content']
        response_text = json.loads(response.text)['choices'][0]['message']['content']
        # except Exception as e:
        #     self.logger.error(f"Error getting response text: {e}")
        #     response_text = "Error getting response text"
        # return "На данных фото изображен кувшин и тарелка"
        return response_text

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
        self.logger.info(f'save_to_chat_history: {chat_id} message_text: {message_text}')
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
            
    def append_file_prefix(self, bot, message_text, message):
        if 'photo' in message or 'document' in message:
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
            file_info = bot.get_file(file_id)
            file_path = file_info.file_path
            self.logger.info(f'file_path: {file_path}')
            message_text = 'files:[' + file_path + ']\n' + message_text
        return message_text

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
    def llm_request(self, bot, user_session, message, system_content=None):
        chat_id = message['chat']['id']
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)

        # Define the first name of the user
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        elif 'username' in message['from']:
            first_name = message['from']['username']
        elif 'id' in message['from']:
            first_name = message['from']['id']
        else:
            first_name = 'Unknown'

        # Check if the message contains text or caption
        if 'text' in message:
            message_text = message['text']
        elif 'caption' in message:
            message_text = message['caption']
        else:
            message_text = ''
            self.logger.error(f'No text or caption in message: {message}')
            if 'photo' in message or 'document' in message:
                message_text = self.append_file_prefix(bot, message_text, message)
                self.save_to_chat_history(
                    message['chat']['id'], 
                    message_text, 
                    message["message_id"],
                    'HumanMessage',
                    message["date"],
                    first_name
                )
            # return 'No text or caption in message'
            return ''
        
        

        # If message contains an attached images
        message_text = self.append_file_prefix(bot, message_text, message)

        self.save_to_chat_history(
            message['chat']['id'], 
            message_text, 
            message["message_id"],
            'HumanMessage',
            message["date"],
            first_name
            )
        
        
            # self.logger.info(f'photo: {message["photo"]}')
            # self.save_to_chat_history(
            #     message['chat']['id'], 
            #     'photo', 
            #     message["message_id"],
            #     'HumanMessage'
            #     )
            # return 'photo'
        # Add the [chat_id] and the [message_id] as a prefix to the message_text
        message_text = f"chat_id: {chat_id}\nmessage_id: {message['message_id']}\n {message_text}"
        self.logger.info(f'invoking message_text: {message_text}')
        response = self.chat_agent.agent_executor.invoke(
            {
                "input": message_text,
                "chat_history": self.chat_history,
            }
        )["output"]
        
        self.save_to_chat_history(
            message['chat']['id'],
            response,
            message["message_id"],
            'AIMessage'
            )

        return response
