#!/usr/bin/env python3
"""Serving check for the Ling-3.0 INT4 checkpoints.

Drives any OpenAI-compatible endpoint: four greedy sanity prompts (factual,
reasoning with thinking on, a tool call, short code), then prompt and decode
throughput at about 1K and 8K tokens. Requires the server's /tokenize and
/detokenize helpers so the prompt lengths are measured, not estimated.

    python bench/serve_check.py http://127.0.0.1:8000/v1 MODEL OUT.json 3
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "auto"
OUT = sys.argv[3] if len(sys.argv) > 3 else "serving.json"
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 3

def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE.replace('/v1', '') + path,
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def chat(payload, timeout=600):
    req = urllib.request.Request(BASE + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0

def stream_chat(payload, timeout=1200):
    req = urllib.request.Request(BASE + '/chat/completions',
        data=json.dumps(dict(payload, stream=True, stream_options={'include_usage': True})).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.time(); first = None; last = None; chunks = 0; text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            obj = json.loads(data)
            if obj.get('usage'):
                usage = obj['usage']
            for ch in obj.get('choices', []):
                d = ch.get('delta', {})
                piece = d.get('content') or d.get('reasoning_content')
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now; chunks += 1; text.append(piece)
    return {'usage': usage, 'ttft': (first - t0) if first else None,
            'decode_s': (last - first) if first and last else None,
            'chunks': chunks, 'text': ''.join(text)[:800], 'wall': time.time() - t0}

# ---- build token-length prompts by slicing detokenized corpus ----
CORPUS = ("The Ling-3.0 flash checkpoint is a mixture-of-experts language model used for agentic "
          "coding, mathematical reasoning and tool use. In a serving test we verify that every "
          "tensor is present, that the shard index resolves, and that greedy decoding is stable. "
          "A careful operator records the prompt and decode throughput on the same runtime so "
          "that any difference can be compared with the run-to-run noise floor. ") * 300

def make_prompt(n_tokens, nonce):
    toks = post('/tokenize', {'model': MODEL, 'prompt': CORPUS, 'add_special_tokens': False})['tokens']
    ids = toks[:max(1, n_tokens - 8)]
    text = post('/detokenize', {'model': MODEL, 'tokens': ids})['prompt']
    return f"Session {nonce}. " + text

result = {'model': MODEL, 'base': BASE, 'repeats': REPEATS, 'tasks': {}, 'speed': []}
nonce0 = ''.join(random.choice(string.ascii_uppercase) for _ in range(8))

# 1 factual, thinking off
body, wall = chat({'model': MODEL, 'messages': [{'role':'user','content':'What is the capital of France? Reply with just the city name.'}],
    'temperature': 0.0, 'max_tokens': 24, 'chat_template_kwargs': {'enable_thinking': False}})
m = body['choices'][0]['message']
result['tasks']['factual'] = {'content': m.get('content'), 'reasoning': (m.get('reasoning_content') or '')[:200],
    'usage': body.get('usage'), 'wall_s': round(wall,2), 'finish': body['choices'][0].get('finish_reason')}

# 2 thinking-on reasoning
q = ("A warehouse has 3 shelves. The first shelf holds 7 boxes. The second shelf holds twice as many "
     "boxes as the first. The third shelf holds 4 fewer boxes than the second. Each box contains 5 "
     "widgets, and 12 widgets are defective. How many non-defective widgets are in the warehouse? "
     "Show your reasoning step by step, then give the final number.")
body, wall = chat({'model': MODEL, 'messages': [{'role':'user','content':q}], 'temperature': 0.0,
    'max_tokens': 700, 'chat_template_kwargs': {'enable_thinking': True}})
m = body['choices'][0]['message']
result['tasks']['reasoning'] = {'content': (m.get('content') or '')[:700],
    'reasoning': (m.get('reasoning_content') or '')[:900],
    'usage': body.get('usage'), 'wall_s': round(wall,2), 'finish': body['choices'][0].get('finish_reason')}

# 3 tool call
tools = [{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city',
    'parameters':{'type':'object','properties':{'city':{'type':'string'},'unit':{'type':'string','enum':['c','f']}},
    'required':['city']}}}]
body, wall = chat({'model': MODEL, 'messages': [{'role':'user','content':"What's the weather in Paris? Use the get_weather tool in celsius."}],
    'temperature': 0.0, 'max_tokens': 300, 'tools': tools, 'tool_choice': 'auto',
    'chat_template_kwargs': {'enable_thinking': False}})
m = body['choices'][0]['message']
result['tasks']['tool'] = {'content': m.get('content'), 'tool_calls': m.get('tool_calls'),
    'usage': body.get('usage'), 'wall_s': round(wall,2), 'finish': body['choices'][0].get('finish_reason')}

# 4 short code
body, wall = chat({'model': MODEL, 'messages': [{'role':'user','content':'Write a Python function `sum_sq(xs)` that returns the sum of the squares of a list of numbers. Return only the code.'}],
    'temperature': 0.0, 'max_tokens': 220, 'chat_template_kwargs': {'enable_thinking': False}})
m = body['choices'][0]['message']
result['tasks']['code'] = {'content': (m.get('content') or '')[:700], 'usage': body.get('usage'),
    'wall_s': round(wall,2), 'finish': body['choices'][0].get('finish_reason')}

# 5 speed 1K and 8K, repeats
for n in (1024, 8192):
    for r in range(REPEATS):
        nonce = nonce0 + ''.join(random.choice(string.ascii_uppercase) for _ in range(6))
        prompt = make_prompt(n, nonce)
        s = stream_chat({'model': MODEL, 'messages':[{'role':'user','content':prompt}],
            'temperature': 0.0, 'max_tokens': 64, 'ignore_eos': True,
            'chat_template_kwargs': {'enable_thinking': False}})
        u = s['usage'] or {}
        rec = {'target': n, 'run': r+1, 'prompt_tokens': u.get('prompt_tokens'),
               'completion_tokens': u.get('completion_tokens'), 'ttft_s': round(s['ttft'],3) if s['ttft'] else None,
               'decode_s': round(s['decode_s'],3) if s['decode_s'] else None,
               'prompt_tps': round(u.get('prompt_tokens',0)/s['ttft'],2) if s['ttft'] else None,
               'decode_tps': round(u.get('completion_tokens',0)/s['decode_s'],2) if s['decode_s'] else None,
               'chunks': s['chunks'], 'wall_s': round(s['wall'],2)}
        result['speed'].append(rec)
        print(json.dumps(rec), flush=True)

json.dump(result, open(OUT,'w'), indent=1)
print('WROTE', OUT)
