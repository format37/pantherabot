"""view_image hands the model an image it can use, in a tool result that fits.

On 2026-10-06 a 930 KB photo ended an answer. The CLI scaled it to 2000 px and
500 KB, and passed it on as one line of JSON that held it twice as base64:
1.33 MB, over the SDK's buffer of 1 MB.
"""
import asyncio
import base64
import io
import sys
import threading
from pathlib import Path

from PIL import Image

from conftest import ALICE, message

SDK_DEFAULT_LINE = 1024 * 1024
# The line of 2026-10-06, as measured: 1 332 974 bytes around two copies of an
# image of 666 248 base64 characters.
LINE_IMAGE_BASE64 = 666_248
LINE_ENVELOPE = 1_332_974 - 2 * LINE_IMAGE_BASE64


def photo(width, height, sigma=20, quality=90, **save):
    """A JPEG that compresses as badly as a detailed photo, or worse."""
    noise = Image.effect_noise((width, height), sigma)
    im = Image.merge('RGB', (noise, noise.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                             noise.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality, **save)
    return buf.getvalue()


def png(width, height, color, mode='RGB'):
    buf = io.BytesIO()
    Image.new(mode, (width, height), color).save(buf, 'PNG')
    return buf.getvalue()


def view(env, name, data, chat_id=ALICE):
    """Put a file in the chat's work dir and look at it as the model would."""
    folder = env.work / str(chat_id)
    folder.mkdir(exist_ok=True)
    (folder / name).write_bytes(data)
    tools = env.bot_tools.build_tools(chat_id, 1, [])
    return asyncio.run(tools['view_image'].handler({'path': name}))


def shown(result):
    """(bytes, mime type, PIL image, note) of a view_image result."""
    assert not result.get('is_error'), result
    image, *rest = result['content']
    assert image['type'] == 'image'
    data = base64.standard_b64decode(image['data'])
    note = rest[0]['text'] if rest else None
    return data, image['mimeType'], Image.open(io.BytesIO(data)), note


def fits(env, result):
    """The result is within the byte budget, and its line within the SDK's default."""
    data, mime, im, _ = shown(result)
    assert mime == 'image/jpeg' and im.format == 'JPEG'
    assert len(data) <= env.bot_tools.VIEW_MAX_BYTES
    assert 2 * len(result['content'][0]['data']) + 10 * LINE_ENVELOPE < SDK_DEFAULT_LINE
    assert max(im.size) <= env.bot_tools.VIEW_MAX_SIDE
    assert im.width * im.height <= env.bot_tools.VIEW_MAX_PIXELS * 1.001
    return im


def test_an_image_that_fits_is_passed_untouched(env):
    plot = png(640, 480, 'navy')
    data, mime, _, note = shown(view(env, 'plot.png', plot))
    assert data == plot and mime == 'image/png' and note is None

    # The type comes from the file, not from its name.
    data, mime, _, note = shown(view(env, 'plot.jpg', plot))
    assert data == plot and mime == 'image/png' and note is None

    small = photo(800, 600)
    data, mime, _, note = shown(view(env, 'photo.jpg', small))
    assert data == small and mime == 'image/jpeg' and note is None


def test_the_photo_that_ended_an_answer_fits(env):
    """2560x1183 and 930 KB, as Telegram stores a photo from a current phone."""
    original = photo(2560, 1183, sigma=14, quality=88)
    assert 850_000 < len(original) < 1_100_000, len(original)

    result = view(env, 'file_1378.jpg', original)
    assert fits(env, result).size == (2000, 924)
    assert shown(result)[3] == 'Shown at 2000x924; the file is 2560x1183.'


def test_an_image_with_few_pixels_and_many_bytes_is_only_re_encoded(env):
    original = photo(1600, 1200, sigma=12, quality=95)
    assert len(original) > 2 * env.bot_tools.VIEW_MAX_BYTES, len(original)
    result = view(env, 'detailed.jpg', original)
    assert fits(env, result).size == (1600, 1200)
    assert len(result['content']) == 1                   # nothing to tell the model


def test_a_big_photo_is_scaled_to_what_the_model_reads(env):
    original = photo(4000, 3000, sigma=8, quality=85)
    assert len(original) < env.bot_tools.MAX_IMAGE_BYTES, len(original)
    result = view(env, 'camera.jpg', original)
    assert fits(env, result).size == (2000, 1500)        # the long edge
    assert shown(result)[3] == 'Shown at 2000x1500; the file is 4000x3000.'

    square = view(env, 'square.jpg', photo(3000, 3000, sigma=8, quality=85))
    assert fits(env, square).size == (1936, 1936)        # 3.75 megapixels


def test_an_image_no_jpeg_quality_can_fit_gets_smaller(env):
    """Pure noise: quality alone does not bring it under the budget."""
    result = view(env, 'noise.jpg', photo(2000, 1130, sigma=80, quality=95))
    im = fits(env, result)
    assert im.width < 2000 and abs(im.width / im.height - 2000 / 1130) < 0.01
    assert im.width * im.height > 500_000                # and no smaller than it must
    assert shown(result)[3] == f'Shown at {im.width}x{im.height}; the file is 2000x1130.'


def test_a_photo_stored_on_its_side_is_shown_upright(env):
    exif = Image.Exif()
    exif[0x0112] = 6                                     # rotate 90 degrees to view
    result = view(env, 'portrait.jpg', photo(3000, 2000, sigma=8, quality=85, exif=exif))
    assert fits(env, result).size == (1333, 2000)
    assert shown(result)[3] == 'Shown at 1333x2000; the file is 2000x3000.'


def test_a_phone_mpo_is_a_jpeg(env):
    first = Image.open(io.BytesIO(photo(3000, 2000, sigma=8, quality=85)))
    buf = io.BytesIO()
    first.save(buf, 'MPO', save_all=True, append_images=[first.resize((300, 200))])
    assert Image.open(io.BytesIO(buf.getvalue())).format == 'MPO'
    assert fits(env, view(env, 'phone.jpg', buf.getvalue())).size == (2000, 1333)


def test_transparency_goes_onto_white(env):
    result = view(env, 'logo.png', png(3000, 3000, (200, 0, 0, 0), mode='RGBA'))
    im = fits(env, result)
    assert im.size == (1936, 1936)
    assert min(im.convert('RGB').getpixel((968, 968))) >= 250


def test_what_cannot_be_viewed_is_an_error_for_the_model(env, monkeypatch):
    result = view(env, 'notes.jpg', b'these are not pixels')
    assert result['is_error'] and 'cannot view notes.jpg: not a JPEG' in result['content'][0]['text']

    result = view(env, 'map.png', png(6000, 5000, 'white'))
    assert result['is_error'] and '6000x5000 is too many pixels' in result['content'][0]['text']

    whole = photo(4000, 3000, sigma=8, quality=85)
    result = view(env, 'cut.jpg', whole[:len(whole) // 2])
    assert result['is_error'] and 'damaged' in result['content'][0]['text']

    monkeypatch.setattr(env.bot_tools, 'MAX_IMAGE_BYTES', len(whole) - 1)
    result = view(env, 'whole.jpg', whole)
    assert result['is_error'] and 'too large to view' in result['content'][0]['text']


def test_decoding_does_not_run_on_the_event_loop(env, monkeypatch):
    threads = []

    def spy(path):
        threads.append(threading.get_ident())
        return b'\xff\xd8\xff', 'image/jpeg', None

    monkeypatch.setattr(env.bot_tools, 'image_for_model', spy)
    view(env, 'any.jpg', b'x')
    assert threads and threads[0] != threading.get_ident()


def test_a_line_over_one_megabyte_does_not_end_the_answer(env, client, monkeypatch, tmp_path):
    """Through the real SDK: a line the size of the one of 2026-10-06 is read."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    state = tmp_path / 'fake-cli'
    state.mkdir()
    wrapper = tmp_path / 'claude'
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {Path(__file__).with_name("fake_claude.py")} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv('FAKE_CLAUDE_DIR', str(state))
    monkeypatch.setenv('FAKE_CLAUDE_IMAGE_BASE64', str(LINE_IMAGE_BASE64))
    monkeypatch.setenv('CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK', '1')
    monkeypatch.setattr(subprocess_cli.SubprocessCLITransport, '_find_cli', lambda self: str(wrapper))
    monkeypatch.delattr(env.server.panthera, '_claude_agent_query')   # the real one

    client.post('/message', json=message(ALICE, 10, 'what is in this photo?'))

    assert (state / 'answered').exists()
    assert [e['text'] for e in env.events] == ['hello from the fake CLI']
