#!/usr/bin/env python3
"""
CLI interface for bot tools, callable from Claude agent via Bash.

Usage:
    python3 /server/tools_cli.py <tool_name> '<json_args>'

Examples:
    python3 /server/tools_cli.py wolfram_alpha '{"query": "solve x^2+1=0"}'
    python3 /server/tools_cli.py web_search '{"query": "latest AI news"}'
    python3 /server/tools_cli.py generate_image '{"prompt": "a cat", "chat_id": 123, "message_id": 456}'
"""
import sys
import json
import os
import asyncio
import requests
import httpx
import telebot
from telebot.formatting import escape_markdown
from google import genai
from google.genai import types
import mimetypes

# Configure Telegram bot API to use local server
server_api_uri = 'http://localhost:8081/bot{0}/{1}'
telebot.apihelper.API_URL = server_api_uri
server_file_url = 'http://localhost:8081'
telebot.apihelper.FILE_URL = server_file_url

with open('config.json') as f:
    config = json.load(f)

bot = telebot.TeleBot(config['TOKEN'])


async def wolfram_alpha(query):
    """Query Wolfram|Alpha for math/science."""
    appid = os.getenv("WOLFRAM_ALPHA_APPID") or os.getenv("WOLFRAM_ALPHA_APP_ID")
    if not appid:
        return "Wolfram|Alpha is not configured: missing WOLFRAM_ALPHA_APPID."

    try:
        params = {
            "appid": appid,
            "input": query,
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
            err = qr.get("error") or {}
            msg = err.get("msg") or qr.get("didyoumeans") or "query was not successful"
            return f"Wolfram|Alpha could not answer: {msg}"

        pods = qr.get("pods", []) or []
        preferred = {"result", "results", "solutions", "solution", "root", "roots",
                     "definite integral", "derivative"}
        lines, tail = [], []
        for pod in pods:
            title = (pod.get("title") or "").strip()
            subpods = pod.get("subpods", []) or []
            texts = [sp.get("plaintext", "").strip() for sp in subpods
                     if sp.get("plaintext", "").strip()]
            if not texts:
                continue
            entry = f"{title}: {texts[0]}" if title else texts[0]
            if title.lower() in preferred:
                lines.append(entry)
            else:
                tail.append(entry)

        output = "\n".join(lines + tail).strip()
        return output if output else "No plaintext results."
    except Exception as e:
        return f"Wolfram|Alpha error: {e}"


async def web_search(query):
    """Search the web using Perplexity Pro."""
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        return json.dumps({"answer": "Web search unavailable: missing PERPLEXITY_API_KEY.",
                           "citations": []})

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant that provides answers with sources."},
                {"role": "user", "content": query},
            ],
            "temperature": 0.5,
            "stream": False,
        }

        resp = requests.post("https://api.perplexity.ai/chat/completions",
                             headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        citations = result.get('citations', []) or []

        return json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"answer": f"Web search failed: {e}", "citations": []})


async def generate_image(prompt, chat_id, message_id, file_list=None):
    """Generate image using Gemini and send to Telegram chat."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Image generation failed: GEMINI_API_KEY not configured"

    client = genai.Client(api_key=api_key)

    try:
        parts = []
        if file_list:
            for file_path in file_list:
                with open(file_path, "rb") as img_file:
                    image_bytes = img_file.read()
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = "image/jpeg"
                parts.append(types.Part.from_bytes(mime_type=mime_type, data=image_bytes))

        parts.append(types.Part.from_text(text=prompt))
        contents = [types.Content(role="user", parts=parts)]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size="1K"),
            tools=[types.Tool(googleSearch=types.GoogleSearch())],
        )

        image_data = None
        text_response = None

        for chunk in client.models.generate_content_stream(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=generate_content_config,
        ):
            if (chunk.candidates and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts):
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        image_data = part.inline_data.data
                    elif part.text:
                        text_response = part.text

        if not image_data:
            return "Image generation failed: No image data returned"

        caption_text = text_response if text_response else prompt
        if len(caption_text) > 1000:
            caption_text = caption_text[:1000]
        caption = f"||{escape_markdown(caption_text)}||"

        sent_message = bot.send_photo(
            chat_id=int(chat_id),
            photo=image_data,
            reply_to_message_id=int(message_id),
            caption=caption,
            parse_mode="MarkdownV2"
        )

        # Save file_id for inline queries
        file_id = sent_message.photo[-1].file_id
        image_dir = f"data/users/{chat_id}/images"
        os.makedirs(image_dir, exist_ok=True)
        with open(os.path.join(image_dir, file_id), 'w') as f:
            f.write("")

        return "Image generated and sent to the chat"
    except Exception as e:
        return f"Image generation failed: {e}"


async def update_system_prompt(chat_id, new_prompt):
    """Update the system prompt for a chat."""
    os.makedirs('./data/custom_prompts', exist_ok=True)
    with open(f'./data/custom_prompts/{chat_id}.txt', 'w') as f:
        f.write(new_prompt)
    return "System prompt updated"


async def reset_system_prompt(chat_id):
    """Reset the system prompt for a chat."""
    path = f'./data/custom_prompts/{chat_id}.txt'
    if os.path.exists(path):
        os.remove(path)
    return "System prompt reset: ok"


TOOLS = {
    "wolfram_alpha": wolfram_alpha,
    "web_search": web_search,
    "generate_image": generate_image,
    "update_system_prompt": update_system_prompt,
    "reset_system_prompt": reset_system_prompt,
}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 tools_cli.py <tool_name> '<json_args>'")
        print(f"Available tools: {', '.join(TOOLS.keys())}")
        sys.exit(1)

    tool_name = sys.argv[1]
    if tool_name not in TOOLS:
        print(f"Unknown tool: {tool_name}. Available: {', '.join(TOOLS.keys())}")
        sys.exit(1)

    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    result = asyncio.run(TOOLS[tool_name](**args))
    print(result)
