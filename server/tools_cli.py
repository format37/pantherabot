#!/usr/bin/env python3
"""
CLI interface for bot tools, callable from Claude agent via Bash.

Usage:
    python3 /server/tools_cli.py <tool_name> '<json_args>'

Examples:
    python3 /server/tools_cli.py wolfram_alpha '{"query": "solve x^2+1=0"}'
    python3 /server/tools_cli.py web_search '{"query": "latest AI news"}'
    python3 /server/tools_cli.py generate_image '{"prompt": "a cat", "chat_id": 123, "message_id": 456}'
    python3 /server/tools_cli.py remember '{"chat_id": 123, "note": "prefers short replies"}'

Every tool that takes a chat_id is pinned to the conversation the request came
from: panthera.py exports PANTHERA_CHAT_ID into the CLI environment and
check_chat_id() below refuses anything else, so a prompt injection cannot write
another chat's memory, prompt or images.
"""
import sys
import json
import os
import asyncio
import httpx
import telebot
from telebot.formatting import escape_markdown
from google import genai
from google.genai import types
import mimetypes
import base64

import memory

# Configure Telegram bot API to use local server
server_api_uri = 'http://localhost:8081/bot{0}/{1}'
telebot.apihelper.API_URL = server_api_uri
server_file_url = 'http://localhost:8081'
telebot.apihelper.FILE_URL = server_file_url

with open('config.json') as f:
    config = json.load(f)

bot = telebot.TeleBot(config['TOKEN'])


def check_chat_id(chat_id):
    """Refuse a chat_id that is not the one this turn belongs to.

    PANTHERA_CHAT_ID is set by panthera.py for every model turn. It is absent
    when an operator runs a tool by hand, and then no restriction applies.
    """
    given = str(chat_id).strip()
    expected = os.environ.get('PANTHERA_CHAT_ID', '').strip()
    if expected and given != expected:
        raise PermissionError(
            f'chat_id {given} does not belong to this conversation ({expected}); refused'
        )
    return given


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


IMAGE_MODEL_VERTEX = "gemini-3-pro-image"          # Nano Banana Pro: the strongest image model on Vertex AI (2026-09-06)
IMAGE_MODEL_DEVELOPER = "gemini-3-pro-image"       # the same model on the Developer API; gemini-3.1-flash-image is the cheaper, faster alternative

# Telegram turns every photo into a JPEG and refuses one over 10 MB. A 4K PNG
# from Gemini is 16–24 MB, so it is encoded here first (see as_telegram_photo).
TELEGRAM_PHOTO_MAX_BYTES = 10 * 1024 * 1024


def genai_client():
    """The Gemini client to use, and the image model name that goes with it.

    Vertex AI when a service account is configured (VERTEX_SA_JSON_B64: the
    key file, base64-encoded), otherwise the Developer API with GEMINI_API_KEY.
    Vertex is the only Google door open from the production host: the
    Developer API refuses that address for every project, paid or free, with
    "User location is not supported", and no billing state changes it
    (measured 2026-09-06). The Developer API path stays for hosts it does
    accept.
    """
    sa_b64 = os.environ.get("VERTEX_SA_JSON_B64", "").strip()
    if sa_b64:
        from google.oauth2 import service_account
        info = json.loads(base64.b64decode(sa_b64))
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = os.environ.get("VERTEX_PROJECT", "").strip() or info.get("project_id")
        location = os.environ.get("VERTEX_LOCATION", "").strip() or "global"
        client = genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        return client, os.environ.get("GEMINI_IMAGE_MODEL", "").strip() or IMAGE_MODEL_VERTEX
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("neither VERTEX_SA_JSON_B64 nor GEMINI_API_KEY is configured")
    return genai.Client(api_key=api_key), os.environ.get("GEMINI_IMAGE_MODEL", "").strip() or IMAGE_MODEL_DEVELOPER


def as_telegram_photo(image_data):
    """Bytes Telegram accepts as a photo: a JPEG when the original is too big.

    Telegram re-encodes photos to JPEG anyway, so nothing is lost; an image
    that already fits is passed through untouched.
    """
    if len(image_data) <= TELEGRAM_PHOTO_MAX_BYTES:
        return image_data
    import io
    from PIL import Image
    with Image.open(io.BytesIO(image_data)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


async def generate_image(prompt, chat_id, message_id, file_list=None):
    """Generate an image with Gemini (nano banana) and send it to the Telegram chat."""
    check_chat_id(chat_id)
    try:
        client, model = genai_client()
    except Exception as e:
        return f"Image generation failed: {e}"

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
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="4K",
            ),
        )

        # Off the event loop: the call takes 40-70 s, and on the loop it froze
        # every chat for that long (2026-09-10..16: the log went silent for
        # 37-42 s per image, and a group message sat unread for 35 s).
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        image_data = None
        text_response = None

        for part in response.candidates[0].content.parts:
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

        # Encoding a 4K PNG and uploading 10 MB are seconds of work as well.
        photo = await asyncio.to_thread(as_telegram_photo, image_data)
        sent_message = await asyncio.to_thread(
            bot.send_photo,
            chat_id=int(chat_id),
            photo=photo,
            reply_to_message_id=int(message_id),
            caption=caption,
            parse_mode="MarkdownV2",
            # The message being answered may be gone by the time an image is
            # ready; losing the image over that is worse than losing the link.
            allow_sending_without_reply=True,
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


async def render_math(formula, chat_id, message_id):
    """Render a LaTeX math formula as PNG image and send to Telegram chat."""
    check_chat_id(chat_id)
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    formula = formula.strip()
    if not (formula.startswith('$') and formula.endswith('$')):
        formula = f'${formula}$'

    try:
        fig = Figure(facecolor='white')
        fig.text(0.5, 0.5, formula, ha='center', va='center',
                 fontsize=18, color='black')
        buf = io.BytesIO()
        fig.savefig(buf, dpi=200, format='png', bbox_inches='tight', pad_inches=0.3)
        buf.seek(0)
        bot.send_photo(
            chat_id=int(chat_id),
            photo=buf,
            reply_to_message_id=int(message_id),
            allow_sending_without_reply=True,
        )
        return "Math formula rendered and sent as image"
    except Exception as e:
        return f"Math rendering failed: {e}"


async def update_system_prompt(chat_id, new_prompt):
    """Update the system prompt for a chat."""
    check_chat_id(chat_id)
    os.makedirs('./data/custom_prompts', exist_ok=True)
    with open(f'./data/custom_prompts/{chat_id}.txt', 'w') as f:
        f.write(new_prompt)
    return "System prompt updated"


async def reset_system_prompt(chat_id):
    """Reset the system prompt for a chat."""
    check_chat_id(chat_id)
    path = f'./data/custom_prompts/{chat_id}.txt'
    if os.path.exists(path):
        os.remove(path)
    return "System prompt reset: ok"


async def remember(chat_id, note):
    """Append one note to this chat's long-term memory."""
    check_chat_id(chat_id)
    return memory.remember(chat_id, note)


async def forget(chat_id, pattern):
    """Drop notes matching `pattern` ('*' clears the whole memory)."""
    check_chat_id(chat_id)
    return memory.forget(chat_id, pattern)


async def replace_memory(chat_id, content):
    """Rewrite the whole memory, for condensing it."""
    check_chat_id(chat_id)
    return memory.replace_memory(chat_id, content)


TOOLS = {
    "wolfram_alpha": wolfram_alpha,
    "generate_image": generate_image,
    "render_math": render_math,
    "update_system_prompt": update_system_prompt,
    "reset_system_prompt": reset_system_prompt,
    "remember": remember,
    "forget": forget,
    "replace_memory": replace_memory,
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
    try:
        result = asyncio.run(TOOLS[tool_name](**args))
    except (PermissionError, ValueError, TypeError) as e:
        # Report refusals and bad arguments as a readable line the model can act
        # on, not a traceback. Non-zero exit keeps it visible as a failure.
        print(f'{tool_name} failed: {e}')
        sys.exit(1)
    print(result)
