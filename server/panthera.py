import os
import logging
import json
import requests
import time
import glob
import json
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import YouTubeSearchTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
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
import tenacity
import httpx
from google import genai
from google.genai import types
import mimetypes

with open('config.json') as config_file:
    config = json.load(config_file)
    bot = telebot.TeleBot(config['TOKEN'])

class TextOutput(BaseModel):
    text: str = Field(description="Text output")

class BotActionType(BaseModel):
    val: str = Field(description="Tool parameter value")

# COMMENTED OUT: Native multimodal support replaces this tool
# class image_context_conversation_args(BaseModel):
#     text_request: str = Field(description="Text request in context of images")
#     file_list: List[str] = Field(description="List of file_id")

class text_file_reader_args(BaseModel):
    file_list: List[str] = Field(description="List of file_id")

class update_system_prompt_args(BaseModel):
    chat_id: str = Field(description="chat_id")
    new_prompt: str = Field(description="new_prompt")

class reset_system_prompt_args(BaseModel):
    chat_id: str = Field(description="chat_id")

class ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    # style: str = Field(description="The style of the generated images. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.")
    chat_id: int = Field(description="chat_id")
    message_id: int = Field(description="message_id")
    file_list: List[str] = Field(default_factory=list, description="Optional list of image file paths to use for editing or composition.")

class BFL_ImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    chat_id: int = Field(description="chat_id")
    message_id: int = Field(description="message_id")
    raw_mode: bool = Field(description="raw_mode Generate less processed, more natural-looking images")

class NanoBananaImagePlotterArgs(BaseModel):
    prompt: str = Field(description="The prompt to generate the image")
    chat_id: int = Field(description="chat_id")
    message_id: int = Field(description="message_id")
    file_list: List[str] = Field(default_factory=list, description="Optional list of image file paths to use for editing or composition.")

class ask_reasoning_args(BaseModel):
    request: str = Field(description="Request for the Reasoning expert")

class perplexity_web_search_args(BaseModel):
    request: str = Field(description="The search query or question to run via Perplexity Pro web search")

class WolframQueryArgs(BaseModel):
    request: str = Field(description="Wolfram|Alpha query, e.g., 'solve x^2 + 2*x^2 + 8 = 0'")

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

def encode_image(image_path, logger=None):
    """
    Encode an image file to base64 format.

    Args:
        image_path: Path to the image file (may include Telegram user prefix)
        logger: Optional logger instance for warning messages

    Returns:
        Base64-encoded image string, or None if file doesn't exist
    """
    # Remove user_name prefix from Telegram file paths
    # Example: '/6014837471:AAE5.../photos/file_2525.jpg' -> '/AAE5.../photos/file_2525.jpg'
    clean_path = re.sub(r'^/[^/]+:', '/', image_path)

    try:
        with open(clean_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        if logger:
            logger.warning(f"Image file not found: {clean_path} (original: {image_path})")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"Error encoding image {clean_path}: {str(e)}")
        return None

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
        
        # model = 'gpt-4o'
        # model = 'gpt-4.5-preview'
        # model = 'gpt-4o-2024-11-20'
        # model = 'o1-preview'
        # model = 'o1-mini'
        model = config.get('primary_model')
        temperature = 0.5
        llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
            model=model,
            # temperature=temperature,
        )

        # if "ANTHROPIC_API_KEY" not in os.environ:
        #     self.logger.error("ANTHROPIC_API_KEY is not set")
        # # model="claude-3-5-sonnet-20240620",
        # temperature = 1.0
        # # model="claude-3-5-sonnet-20241022"
        # model="claude-3-7-sonnet-20250219"
        # llm = ChatAnthropic(
        #     model=model,
        #     temperature = temperature
        #     # max_tokens=8192,
        # )        

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

        # Robust Wolfram|Alpha tool via JSON API to avoid brittle XML client assertion
        wolfram_tool = StructuredTool.from_function(
            coroutine=self.wolfram_alpha_json,
            name="wolfram_alpha",
            description=(
                "Query Wolfram|Alpha for math/science. Provide a concise query, e.g., "
                "'solve x^2+2*x^2+8=0'. Returns plaintext results with key pods."
            ),
            args_schema=WolframQueryArgs,
            handle_tool_error=True,
        )
        
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wikipedia_tool = Tool(
                name="wikipedia_search",
                func=wikipedia.run,
                description="Useful when users request biographies or historical moments. Provide links if possible.",
            )

        # COMMENTED OUT: Replaced with native multimodal support
        # OpenAI GPT models now support images directly in chat history
        # coroutine=self.image_context_conversation, # may be used instead of func
        # image_context_conversation_tool = StructuredTool.from_function(
        #     # func=self.image_context_conversation,
        #     coroutine=self.image_context_conversation,
        #     name="image_context_conversation",
        #     description="Answering on your text request about provided images",
        #     args_schema=image_context_conversation_args,
        #     return_direct=False,
        #     handle_tool_error=True,
        #     handle_validation_error=True,
        #     verbose=True,
        # )
        
        bfl_tool_description = """A tool to generate and send to user images based on a given prompt.
1. Be Specific and Descriptive
Instead of "A portrait of a woman," use "A close-up portrait of a middle-aged woman with curly red hair, green eyes, wearing a blue silk blouse."
2. Use Artistic References
Reference specific artists or styles: "Create an image in the style of Van Gogh's 'Starry Night,' but with a futuristic cityscape."
3. Specify Technical Details
Include camera settings and angles: "Wide-angle lens (24mm) at f/1.8, shallow depth of field, focus on subject."
4. Add Mood and Atmosphere
Describe emotional tone: "Cozy, warmly lit bookstore cafe on a rainy evening."
5. Use Contrast and Perspective
- Combine contrasting elements for visual impact
- Experiment with unique viewpoints (bird's-eye, worm's-eye)
- Mix different themes or time periods
Tips:
- Write prompts in natural language
- Describe specific lighting conditions
- Include details about materials and textures
- Be clear about text placement and styling when needed
"""

        image_plotter_tool = StructuredTool.from_function(
            coroutine=self.BFL_ImagePlotterTool,
            name="image_plotter",
            description=bfl_tool_description,
            args_schema=BFL_ImagePlotterArgs,
            # return_direct=False,
        )

        image_plotter_openai_tool = StructuredTool.from_function(
            coroutine=self.ImagePlotterTool_openai,
            name="image_plotter_openai",
            description="Generate image using OpenAI image model and send back to user.",
            args_schema=ImagePlotterArgs,
            # return_direct=False,
        )

        image_plotter_nanobanana_tool = StructuredTool.from_function(
            coroutine=self.ImagePlotterTool_nanobanana,
            name="image_plotter_nanobanana",
            description="Generate image using Gemini (NanoBanana) model and send back to user. Supports multiple input images and text prompt.",
            args_schema=NanoBananaImagePlotterArgs,
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
        # tools.append(youtube_tool)
        # tools.append(google_search_tool)
        # tools.append(wikipedia_tool)
        # COMMENTED OUT: Native multimodal support replaces this tool
        # tools.append(image_context_conversation_tool)
        # tools.append(image_plotter_tool)
        # tools.append(image_plotter_openai_tool)
        tools.append(image_plotter_nanobanana_tool)
        # tools.append(text_file_reader_tool)
        tools.append(update_system_prompt_tool)
        tools.append(reset_system_prompt_tool)
        # tools.append(ask_reasoning_tool)

        # Perplexity web search tool
        perplexity_tool = StructuredTool.from_function(
            coroutine=self.perplexity_web_search,
            name="web_search",
            description=(
                "Performs Perplexity Pro web search and summarizes results with cited sources. "
                "Use when you need up-to-date information from the web."
            ),
            args_schema=perplexity_web_search_args,
        )
        tools.append(perplexity_tool)

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
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         # ("system", system_prompt),
        #         ("system", "{system_prompt}"),
        #         ("placeholder", "{chat_history}"),
        #         ("human", "{input}"),
        #         ("placeholder", "{agent_scratchpad}"),
        #     ]
        # )
        if 'o1' in model:
            prompt_messages = [
                ("system", "{system_prompt}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        else:
            prompt_messages = [
                ("system", "{system_prompt}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    async def wolfram_alpha_json(self, request: str) -> str:
        """
        Call Wolfram|Alpha v2 query API using JSON output and return plaintext pods.
        Requires env var WOLFRAM_ALPHA_APPID or WOLFRAM_ALPHA_APP_ID.
        """
        appid = os.getenv("WOLFRAM_ALPHA_APPID") or os.getenv("WOLFRAM_ALPHA_APP_ID")
        if not appid:
            return "Wolfram|Alpha is not configured: missing WOLFRAM_ALPHA_APPID."

        try:
            params = {
                "appid": appid,
                "input": request,
                "output": "json",
                "format": "plaintext",
                "reinterpret": "true",
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get("https://api.wolframalpha.com/v2/query", params=params)
                resp.raise_for_status()
                data = resp.json()

            qr = data.get("queryresult", {})
            if not qr.get("success"):
                # Try to surface a meaningful message
                err = qr.get("error") or {}
                msg = err.get("msg") or qr.get("didyoumeans") or "query was not successful"
                return f"Wolfram|Alpha could not answer: {msg}"

            pods = qr.get("pods", []) or []
            # Prefer key result pods early
            preferred = {"result", "results", "solutions", "solution", "root", "roots", "definite integral", "derivative"}
            lines = []
            tail = []
            for pod in pods:
                title = (pod.get("title") or "").strip()
                subpods = pod.get("subpods", []) or []
                texts = [sp.get("plaintext", "").strip() for sp in subpods]
                texts = [t for t in texts if t]
                if not texts:
                    continue
                entry = f"{title}: {texts[0]}" if title else texts[0]
                if title.lower() in preferred:
                    lines.append(entry)
                else:
                    tail.append(entry)

            output = "\n".join(lines + tail).strip()
            return output if output else "No plaintext results."
        except httpx.HTTPError as e:
            self.logger.error(f"Wolfram|Alpha HTTP error: {e}")
            return "Wolfram|Alpha request failed due to a network error."
        except Exception as e:
            self.logger.error(f"Wolfram|Alpha unexpected error: {e}", exc_info=True)
            return "Wolfram|Alpha request failed due to an unexpected error."

    async def perplexity_web_search(self, request: str) -> Dict[str, Any]:
        """
        Performs Perplexity Pro web search and returns an LLM composed answer with cited sources.

        Parameters:
            request (str): The search query or question.

        Returns:
            dict with keys:
                - answer (str): The response generated by Perplexity AI based on the query.
                - citations (list[str]): URLs of sources referenced in the answer.

        Note:
            The results of research can be:
            * Outdated
            * Hallucinated
            * Based on generated by LLM articles
        """
        try:
            self.logger.info(f">> Web search: {request}")
            api_key = os.getenv('PERPLEXITY_API_KEY')
            if not api_key:
                self.logger.error("PERPLEXITY_API_KEY is not set")
                return {
                    "answer": "Web search unavailable: missing PERPLEXITY_API_KEY.",
                    "citations": []
                }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            data = {
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides answers with sources."},
                    {"role": "user", "content": request},
                ],
                "temperature": 0.5,
                "stream": False,
            }

            resp = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()

            answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = result.get('citations', []) or []

            # Ensure citations is a list of strings
            if not isinstance(citations, list):
                citations = []

            return {
                "answer": answer,
                "citations": citations,
            }
        except requests.HTTPError as e:
            self.logger.error(f"Perplexity HTTP error: {e} | body={getattr(e.response, 'text', '')}")
            return {
                "answer": "Web search failed due to an HTTP error.",
                "citations": []
            }
        except requests.RequestException as e:
            self.logger.error(f"Perplexity request error: {e}")
            return {
                "answer": "Web search failed due to a network error.",
                "citations": []
            }
        except Exception as e:
            self.logger.error(f"Perplexity unexpected error: {e}", exc_info=True)
            return {
                "answer": "Web search failed due to an unexpected error.",
                "citations": []
            }

    def bfl_generate_image(self, headers, prompt, width=1024, height=768, raw_mode=False):
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
            "raw": raw_mode,
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
            self.logger.info(f"Image URL: {image_url}")
            # Download the image
            image_data = requests.get(image_url).content
            
            # Save the image to the user's images directory
            image_dir = f"data/users/{chat_id}/images"
            os.makedirs(image_dir, exist_ok=True)
            # image_path = os.path.join(image_dir, f"{int(time.time())}.jpg")
            # with open(image_path, 'wb') as f:
            #     f.write(image_data)

            # For Telegram bot API's send_photo method, the caption length limit is 1024 characters. This applies to both regular text and MarkdownV2-formatted captions.
            prompt = prompt[:1000]

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
    
    async def ImagePlotterTool_openai(self, prompt: str, chat_id: int, message_id: int, file_list: List[str] = None) -> str:
        from openai import APIConnectionError, RateLimitError, APIStatusError
        import base64
        from io import BytesIO

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        try:
            image_data = None
            revised_prompt = None
            
            if file_list and len(file_list) > 0:
                # Image editing mode - combine or edit images
                self.logger.info(f"Using image editing with {len(file_list)} images")
                
                # Open all image files
                multiple_images = []
                for file_path in file_list:
                    multiple_images.append(open(file_path, "rb"))
                    # base64_image = encode_image(file_path)
                    # image_url = f"data:image/jpeg;base64,{base64_image}"
                    # multiple_images.append(image_url)
                    # with open(file_path, "rb") as img_file:
                    #     images.append(img_file.read())
                    self.logger.info(f"+ image: {file_path}")
                
                # Use images.edit when images are provided
                edit_response = client.images.edit(
                    model="gpt-image-1",
                    # image=[BytesIO(img) for img in images[:10]],  # Max 10 images
                    # image = multiple_images,
                    image = multiple_images,
                    prompt=prompt,
                )
                
                image_data_obj = edit_response.data[0]
                b64_json = image_data_obj.b64_json
                revised_prompt = getattr(image_data_obj, "revised_prompt", prompt)
                image_data = base64.b64decode(b64_json)
            else:
                # Standard image generation
                self.logger.info("Using standard image generation")
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    user=chat_id
                )
                
                image_data_obj = response.data[0]
                b64_json = image_data_obj.b64_json
                revised_prompt = getattr(image_data_obj, "revised_prompt", prompt)
                image_data = base64.b64decode(b64_json)
            
            if not image_data:
                self.logger.error("ImagePlotterTool: Failed to generate image data")
                return "Image generation failed: No image data returned by OpenAI."
                
            # Use revised_prompt if available, else fallback to original prompt
            caption_text = revised_prompt if revised_prompt else prompt
            caption = f"||{escape_markdown(caption_text)}||"
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
            
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            self.logger.error(f"ImagePlotterTool: OpenAI API error: {e}")
            return f"Image generation failed: OpenAI API error: {str(e)}"
        except Exception as e:
            self.logger.error(f"ImagePlotterTool: Unexpected error: {e}", exc_info=True)
            return "Image generation failed due to an unexpected error. Please try again later."

    async def ImagePlotterTool_nanobanana(self, prompt: str, chat_id: int, message_id: int, file_list: List[str] = None) -> str:
        log_message = f"==> ImagePlotterTool_nanobanana:\nprompt: {prompt}\nfile_list:\n"
        for file_path in file_list or []:
            log_message += f" - {file_path}\n"

        self.logger.info(log_message)

        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("GEMINI_API_KEY not found in environment variables")
            return "Image generation failed: GEMINI_API_KEY not configured"

        client = genai.Client(api_key=api_key)

        try:
            self.logger.info("NanoBanana: Starting image generation request")

            # Build content parts
            parts = []

            # Add images if provided
            if file_list and len(file_list) > 0:
                self.logger.info(f"NanoBanana: Using {len(file_list)} input images")
                for file_path in file_list:
                    # file_path_cropped = re.sub(r'^/[^/]+:', '/', file_path)
                    file_path_cropped = file_path  # Assuming file_path is already clean
                    with open(file_path_cropped, "rb") as img_file:
                        image_bytes = img_file.read()

                    # Detect MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    if not mime_type or not mime_type.startswith('image/'):
                        mime_type = "image/jpeg"  # Default fallback

                    parts.append(
                        types.Part.from_bytes(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    )
                    self.logger.info(f"+ image: {file_path_cropped} (mime: {mime_type})")

            # Add text prompt
            parts.append(types.Part.from_text(text=prompt))

            # Build content
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]

            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size="1K",
                ),
                tools=[types.Tool(googleSearch=types.GoogleSearch())],
            )

            # Call API (streaming)
            self.logger.info("NanoBanana: Submitting request to Gemini API...")
            image_data = None
            text_response = None

            for chunk in client.models.generate_content_stream(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=generate_content_config,
            ):
                if (chunk.candidates and
                    chunk.candidates[0].content and
                    chunk.candidates[0].content.parts):

                    for part in chunk.candidates[0].content.parts:
                        # Check for image data
                        if part.inline_data and part.inline_data.data:
                            image_data = part.inline_data.data
                            self.logger.info("NanoBanana: Image data received")
                        # Check for text response
                        elif part.text:
                            text_response = part.text
                            self.logger.info(f"NanoBanana: Text response: {part.text}")

            if not image_data:
                self.logger.error("NanoBanana: No image data returned from API")
                return "Image generation failed: No image data returned by Gemini"

            # Prepare caption (use text response if available, otherwise use prompt)
            caption_text = text_response if text_response else prompt
            # Truncate if too long (Telegram limit is 1024 characters)
            if len(caption_text) > 1000:
                caption_text = caption_text[:1000]
            caption = f"||{escape_markdown(caption_text)}||"
            self.logger.info(f"NanoBanana: Caption: {caption}")

            # Send the photo via Telegram
            bot.send_photo(
                chat_id=chat_id,
                photo=image_data,
                reply_to_message_id=message_id,
                caption=caption,
                parse_mode="MarkdownV2"
            )

            self.logger.info("NanoBanana: Image sent successfully")
            return "Image generated and sent to the chat"

        except Exception as e:
            self.logger.error(f"NanoBanana: API error: {e}", exc_info=True)
            return f"Image generation failed: {str(e)}"

    async def ImagePlotterTool_openai_dalle(self, prompt: str, style: str, chat_id: int, message_id: int) -> str:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        style = style.lower()
        if style not in ["vivid", "natural"]:
            self.logger.info(f"Style {style} is not supported. Using default style: vivid")
            style = "vivid"

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            user=chat_id
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
    
    # COMMENTED OUT: Replaced with native multimodal support
    # OpenAI GPT models now support images directly in chat history
    # async def image_context_conversation(self, text_request: str, file_list: List[str]):
    #     self.logger.info(f"image_context_conversation request: {text_request}; file_list: {file_list}")
    #     messages = []
    #     for file_path in file_list:
    #         # Remove user_name prefix
    #         # Example: '/6014837471:AAE5.../photos/file_2525.jpg' -> '/AAE5.../photos/file_2525.jpg'
    #         file_path = re.sub(r'^/[^/]+:', '/', file_path)
    #         self.logger.info(f"file_path: {file_path}")
    #         base64_image = encode_image(file_path)
    #         image_url = f"data:image/jpeg;base64,{base64_image}"
    #         append_message(
    #             messages,
    #             "user",
    #             text_request,
    #             image_url
    #         )
    #     api_key = os.environ.get('OPENAI_API_KEY', '')
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {api_key}"
    #     }
    #     model = config.get('primary_model')
    #
    #     response = requests.post(
    #         "https://api.openai.com/v1/chat/completions",
    #         headers=headers,
    #         json={
    #             "model": model,
    #             "messages": messages,
    #             # "max_tokens": 2000
    #         }
    #     )
    #     self.logger.info(f"image_context_conversation response text: {response.text}")
    #     response_text = json.loads(response.text)['choices'][0]['message']['content']
    #     return response_text
    #     # return "Это кот"
    
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
        # Force model override regardless of stored config
        self.config['model'] = config.get('primary_model')
        self.logger.info(f'Overriding default config model to: {self.config["model"]}')

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
        # Force model override in the user session and persist
        primary_model = config.get('primary_model')
        if session.get('model') != primary_model:
            session['model'] = primary_model
            self.logger.info(f'Overriding user session model to: {session["model"]}')
            self.save_user_session(user_id, session)
        # Return the user json file as dict
        return session

    def reset_chat(self, chat_id):
        self.logger.info(f'reset_chat: {chat_id}')
        # Path for user's chat history
        chat_path = os.path.join('data', 'users', str(chat_id), 'chats', str(chat_id))
        # Create folder if not exist
        Path(chat_path).mkdir(parents=True, exist_ok=True)
        # Remove all files in chat path
        for f in os.listdir(chat_path):
            self.logger.info(f'remove file: {f}')
            os.remove(os.path.join(chat_path, f))

    def token_counter(self, text):
        model_for_tokens = self.config.get('model', 'gpt-4o')
        try:
            enc = tiktoken.encoding_for_model(model_for_tokens)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
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
        name_of_user='AI',
        image_paths=None
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
                "text": f"{message_text}",
                "images": image_paths or []
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
        """
        Extract file paths from a Telegram message.

        Returns:
            list: List of file paths, or empty list if no files present
        """
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
                return [file_path]  # Return as list
        return []  # Return empty list instead of empty string

    def read_chat_history(self, chat_id: str):
        '''Reads the chat history from a folder with improved message limit handling.'''
        user_id = chat_id  # Assuming chat_id corresponds to user_id in private chats
        chat_log_path = os.path.join('data', 'users', str(user_id), 'chats', str(chat_id))
        if not os.path.exists(chat_log_path):
            return

        self.chat_history = []
        
        # First, get all files and sort them by creation time (newest first)
        files = []
        for log_file in os.listdir(chat_log_path):
            file_path = os.path.join(chat_log_path, log_file)
            try:
                files.append((file_path, os.path.getctime(file_path)))
            except Exception as e:
                self.logger.error(f'Error getting file creation time: {e}')
                continue
        
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize counters
        message_count = 0
        token_count = 0
        MAX_MESSAGES = 2040  # Setting a conservative limit well below OpenAI's 2048
        MAX_TOKENS = self.config['token_limit'] if 'token_limit' in self.config else 4000

        # Process files from newest to oldest
        for file_path, _ in files:
            if message_count >= MAX_MESSAGES:
                # Remove older files that won't be used
                try:
                    os.remove(file_path)
                    self.logger.info(f'Removed old chat history file: {file_path}')
                except Exception as e:
                    self.logger.error(f'Error removing file: {e}')
                continue

            try:
                with open(file_path, 'r') as file:
                    message = json.load(file)
                    
                    # Calculate tokens for this message
                    message_tokens = self.token_counter(message['text']) if hasattr(self, 'token_counter') else len(message['text'])
                    
                    # Check if adding this message would exceed token limit
                    if token_count + message_tokens > MAX_TOKENS:
                        # Remove the file if it's too old to be included
                        try:
                            os.remove(file_path)
                            self.logger.info(f'Removed file exceeding token limit: {file_path}')
                        except Exception as e:
                            self.logger.error(f'Error removing file: {e}')
                        continue
                    
                    # Add message to history
                    if message['type'] == 'AIMessage':
                        self.chat_history.insert(0, AIMessage(content=message['text']))
                    elif message['type'] == 'HumanMessage':
                        # Check if message has images - reconstruct multimodal content
                        if 'images' in message and message['images']:
                            content_parts = [{"type": "text", "text": message['text']}]
                            for image_path in message['images']:
                                base64_image = encode_image(image_path, self.logger)
                                if base64_image:  # Only add if image was successfully encoded
                                    self.logger.info(f'# read_chat_history: Adding image to chat_history: {image_path}')
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "low"
                                        }
                                    })
                            self.chat_history.insert(0, HumanMessage(content=content_parts))
                        else:
                            # Simple text message (backward compatible)
                            self.chat_history.insert(0, HumanMessage(content=message['text']))

                    message_count += 1
                    token_count += message_tokens

            except Exception as e:
                self.logger.error(f'Error reading chat history file {file_path}: {e}')
                # Remove corrupted file
                try:
                    os.remove(file_path)
                    self.logger.error(f'Removed corrupted file: {file_path}')
                except Exception as remove_error:
                    self.logger.error(f'Error removing corrupted file: {remove_error}')

        self.logger.info(f'Loaded {message_count} messages with {token_count} tokens for chat {chat_id}')

    def read_chat_history_x(self, chat_id: str):
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
    async def llm_request(self, chat_id, message_id, message_text, image_paths=None):
        # message_text may have augmentations
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id)

        # Construct input - multimodal if images present, text otherwise
        if image_paths:
            self.logger.info(f'Creating multimodal input with {len(image_paths)} images')
            content_parts = [{"type": "text", "text": message_text}]
            for image_path in image_paths:
                base64_image = encode_image(image_path, self.logger)
                if base64_image:  # Only add if image was successfully encoded
                    self.logger.info(f'# llm_request: Adding image to input: {image_path}')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    })
            agent_input = content_parts
        else:
            # Keep simple text input for backward compatibility
            agent_input = message_text

        # Add retries and error handling
        try:
            # Using tenacity to retry with exponential backoff
            @tenacity.retry(
                stop=tenacity.stop_after_attempt(3),  # Try 3 times
                wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),  # Wait between retries
                retry=tenacity.retry_if_exception_type((httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout)),
                reraise=True,
                before_sleep=lambda retry_state: self.logger.info(f"API call failed, retrying in {retry_state.next_action.sleep} seconds...")
            )
            async def execute_with_retry():
                return await self.chat_agent.agent_executor.ainvoke(
                    {
                        "input": agent_input,
                        "chat_history": self.chat_history,
                        "system_prompt": system_prompt,
                    }
                )
            
            result = await execute_with_retry()
            self.logger.info(f'result: {str(result)[:100]}...')
            response = result["output"]
            
            # if response is a list
            if isinstance(response, list):
                if len(response) > 0:  # Check if list has elements before accessing
                    response = response[0]
                else:
                    self.logger.info(f'llm_request received empty list response')
                    response = ''
                    
                # if response is a dict
                if isinstance(response, dict):
                    try:
                        response = response['text']
                    except:
                        self.logger.info(f'llm_request response has no "text": {str(response)[:100]}...')
                        response = str(response)

            self.logger.info(f'llm_request response type: {type(response)}')
            self.logger.info(f'llm_request response: {response}')
            
            self.save_to_chat_history(
                chat_id,
                response,
                message_id,
                'AIMessage'
            )

            return response
            
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            error_message = f"I apologize, but I couldn't complete your request due to a connection issue with my backend services. Error: {str(e)}"
            self.logger.error(f"API connection error: {str(e)}")
            
            # Save the error to chat history so we have a record
            self.save_to_chat_history(
                chat_id,
                error_message,
                message_id,
                'AIMessage'
            )
            
            return error_message
            
        except Exception as e:
            error_message = f"I encountered an unexpected error while processing your request. Please try again later."
            self.logger.error(f"Unexpected error in llm_request: {str(e)}", exc_info=True)
            
            # Save the error to chat history
            self.save_to_chat_history(
                chat_id,
                error_message,
                message_id,
                'AIMessage'
            )
            
            return error_message

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
                model="gpt-5-nano",
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
