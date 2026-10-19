#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for the measured fast Ling-3.1 EXL3 path.

One model, one request at a time, 127.0.0.1 only. The point is to give agents
and showcases the same loader settings as the published benchmark instead of
examples/chat.py's autosplit load. See docs/INSTALL.md for the measured flags.

    python bench/serve_fast.py -m "$MODEL" -dev cuda:0 -mode ling \
      -mcl 62 -mct 8 -gcs 8192 -cs 9216 --port 8000

Then POST to /v1/chat/completions. Non-streaming responses only; greedy by
default (the request's temperature is honoured when > 0). Thinking is off unless
the request sets "thinking": true (also accepted as "think" or in
"chat_template_kwargs").
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def load_ling_format(exllamav3_dir):
    path = Path(exllamav3_dir) / "examples" / "chat_templates.py"
    spec = importlib.util.spec_from_file_location("_ling_chat_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prompt_formats["ling"]("User", "Assistant")


def messages_to_pairs(messages):
    if not messages:
        raise ValueError("messages is empty")
    if messages[-1].get("role") != "user":
        raise ValueError("the last message must have role 'user'")
    system = None
    pairs = []
    user = None
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        if role == "system":
            system = content if system is None else system + "\n" + content
        elif role == "user":
            if user is not None:
                raise ValueError("consecutive user messages are not supported")
            user = content
        elif role == "assistant":
            if user is None:
                raise ValueError("assistant message before a user message")
            pairs.append((user, content))
            user = None
        else:
            raise ValueError(f"unsupported role {role!r}")
    if user is None:
        raise ValueError("the last message must have role 'user'")
    pairs.append((user, None))
    return system, pairs


def want_thinking(body):
    if "thinking" in body:
        return bool(body["thinking"])
    if "think" in body:
        return bool(body["think"])
    kwargs = body.get("chat_template_kwargs") or {}
    return bool(kwargs.get("thinking", kwargs.get("think", False)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="default completion cap when the request omits max_tokens")
    from exllamav3 import model_init
    model_init.add_args(ap, cache=True, add_sampling_args=False,
                        default_cache_size=8192, default_autosplit_max_batch_size=1)
    ap.add_argument("-gcs", "--generator_chunk_size", type=int, default=2048)
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind beyond localhost; use --host 127.0.0.1")
    if args.device is None:
        print("WARNING: no -dev/--device; this is the slower autosplit path", file=sys.stderr)

    import torch
    import exllamav3
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler.presets import ArgmaxSampler, CategoricalSampler

    t0 = time.perf_counter()
    model, config, cache, tokenizer, *_ = model_init.init(args)
    torch.cuda.synchronize()
    print(f"loaded in {time.perf_counter() - t0:.1f} s on {torch.cuda.get_device_name(0)}",
          file=sys.stderr, flush=True)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1,
                    max_chunk_size=args.generator_chunk_size, ngram_match_min=0,
                    recurrent_cache_size=0)
    fmt = load_ling_format(Path(exllamav3.__file__).resolve().parents[1])
    stop = [s for s in fmt.stop_conditions(tokenizer) if s]
    if config.eos_token_id_list and all(config.eos_token_id_list):
        stop += config.eos_token_id_list
    if args.no_warmup:
        # model_init skipped its post-load warm-up. Pay the generator's compile
        # and autotune cost on a throwaway request, not the first user request.
        text = fmt.format(fmt.default_system_prompt(False), [("Hello", None)], False)
        ids = tokenizer.encode(text, add_bos=fmt.add_bos(), encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=8,
                  stop_conditions=stop, sampler=ArgmaxSampler(), decode_special_tokens=False)
        gen.enqueue(job)
        with torch.inference_mode():
            while gen.num_remaining_jobs():
                for _ in gen.iterate():
                    pass
        print("generator warm-up done", file=sys.stderr, flush=True)
    lock = threading.Lock()
    model_name = Path(args.model_dir).name

    def complete(body):
        think = want_thinking(body)
        system, pairs = messages_to_pairs(body.get("messages") or [])
        system_prompt = system or fmt.default_system_prompt(think)
        text = fmt.format(system_prompt, pairs, think)
        ids = tokenizer.encode(text, add_bos=fmt.add_bos(), encode_special_tokens=True)
        prompt_len = ids.shape[1]
        requested = body.get("max_tokens", body.get("max_completion_tokens", args.max_tokens))
        try:
            max_new = int(requested)
        except (TypeError, ValueError):
            raise ValueError("max_tokens must be an integer")
        room = cache.max_num_tokens - prompt_len
        if max_new < 1 or room < 1:
            raise ValueError(f"prompt {prompt_len} tokens leaves no room in a {cache.max_num_tokens}-token cache")
        max_new = min(max_new, room)
        temperature = float(body.get("temperature", 0) or 0)
        sampler = ArgmaxSampler() if temperature == 0 else CategoricalSampler(temperature)
        job = Job(input_ids=ids, max_new_tokens=max_new,
                  stop_conditions=stop, sampler=sampler, decode_special_tokens=False)
        with lock:
            gen.enqueue(job)
            torch.cuda.synchronize()
            final = None
            with torch.inference_mode():
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        if r.get("eos"):
                            final = r
        if final is None:
            raise RuntimeError("generation produced no final result")
        out_ids = job.sequences[0].sequence_ids.torch()[0, prompt_len:].tolist()
        answer = tokenizer.decode(torch.tensor(out_ids, dtype=torch.long), decode_special_tokens=False)
        prefill = float(final["time_prefill"])
        decode = float(final["time_generate"])
        new = int(final["new_tokens"])
        timings = {
            "prefill_s": prefill,
            "decode_s": decode,
            "prefill_tps": prompt_len / prefill if prefill > 0 else None,
            "decode_tps": (new - 1) / decode if new > 1 and decode > 0 else None,
            "ttft_s": prefill,
            "prompt_tokens": int(final["prompt_tokens"]),
            "completion_tokens": new,
            "cached_tokens": int(final["cached_tokens"]),
            "thinking": think,
        }
        print(json.dumps({"event": "request", **timings}), file=sys.stderr, flush=True)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": new,
                      "total_tokens": prompt_len + new},
            "timings": timings,
        }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "local"}]})
            elif self.path in ("/health", "/"):
                self._send(200, {"status": "ok", "model": model_name})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if body.get("stream"):
                    raise ValueError("streaming is not implemented; send stream=false")
                self._send(200, complete(body))
            except Exception as e:
                self._send(400, {"error": {"message": f"{type(e).__name__}: {e}"}})

    print(f"serving {model_name} on http://{args.host}:{args.port}/v1", file=sys.stderr, flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
