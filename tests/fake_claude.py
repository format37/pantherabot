#!/usr/bin/env python3
"""A stand-in for the bundled `claude` CLI: just enough stream-json for the SDK.

The first run in a directory hangs mid-turn and ignores stdin EOF, as CLI
2.1.259 does while a model call is in flight, so only the SDK's SIGTERM ends
it; later runs answer at once. FAKE_CLAUDE_DIR holds the state: a pid file per
run, and `answered` once a run has answered.
"""
import json
import os
import sys
import time

if '-v' in sys.argv:
    print('2.1.259 (Claude Code)')
    sys.exit(0)

state = os.environ['FAKE_CLAUDE_DIR']
runs = sorted(n for n in os.listdir(state) if n.endswith('.pid'))
with open(os.path.join(state, f'{len(runs):02d}.pid'), 'w') as f:
    f.write(str(os.getpid()))
hang = not runs


def send(obj):
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    if msg.get('type') == 'control_request' and msg['request']['subtype'] == 'initialize':
        send({'type': 'control_response', 'response': {
            'subtype': 'success', 'request_id': msg['request_id'], 'response': {}}})
    elif msg.get('type') == 'user' and not hang:
        send({'type': 'assistant', 'message': {
            'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'fake',
            'content': [{'type': 'text', 'text': 'hello from the fake CLI'}],
            'stop_reason': 'end_turn', 'usage': {'input_tokens': 1, 'output_tokens': 1}},
            'parent_tool_use_id': None, 'session_id': 's1', 'uuid': 'u1'})
        send({'type': 'result', 'subtype': 'success', 'is_error': False,
              'duration_ms': 1, 'duration_api_ms': 1, 'num_turns': 1, 'result': 'ok',
              'session_id': 's1', 'total_cost_usd': 0, 'usage': {}, 'uuid': 'u2'})
        open(os.path.join(state, 'answered'), 'w').close()
if hang:
    while True:
        time.sleep(1)
