#!/usr/bin/env python3
"""Screen 8K code greedy repeatability with one model load (R9700 configuration).

Two default passes and two deterministic passes, each 256 tokens, with a fresh
Generator each time. Both arms get an excluded 16-token warm-up. Writes actual
ids as little-endian uint32 .bin files next to --out, plus hashes and timings.
The deterministic arm switches the upstream tuning objects that the three
EXL3_MOE_* environment variables initialize, including already-created stream
states. This permits a same-load A/B; it does not modify the engine checkout.
"""
import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--exllamav3', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    for k in ('EXL3_MOE_STREAM_T', 'EXL3_MOE_STREAM_FUSED_T', 'EXL3_MOE_RECON_DET'):
        if k in os.environ:
            raise SystemExit(f'Unset {k} for the default control')
    sys.path.insert(0, str(Path(a.exllamav3).resolve()))
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    from exllamav3.model import moe_cpu_host
    from exllamav3.modules import moe_batch_recon

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = (Path(a.inputs) / 'code-8k.bin').read_bytes()
    assert len(raw) == 8192 * 4
    prompt = list(struct.unpack('<8192I', raw))
    ids = torch.tensor([prompt], dtype=torch.long)
    cfg = Config.from_directory(a.model)
    cfg.infer_params.moe_cpu_offload = 60
    cfg.infer_params.moe_cpu_threads = 8
    tok = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=9216, max_batch_size=1, max_history=0)
    start = time.perf_counter()
    print(json.dumps({'event': 'load_start'}), flush=True)
    model.load(device=None, progressbar=False, max_chunk_size=4096, max_output_size=1)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    print(json.dumps({'event': 'loaded', 'load_seconds': load_s}), flush=True)

    def run_one(n):
        gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1,
                        max_chunk_size=4096, ngram_match_min=0, recurrent_cache_size=0)
        job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[],
                  sampler=ArgmaxSampler(), decode_special_tokens=True)
        final = None
        try:
            gen.enqueue(job)
            with torch.inference_mode():
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        if r.get('eos'):
                            final = r
            seq = job.sequences[0].sequence_ids.torch()[0, 8192:].tolist()
            assert len(seq) == n and final['new_tokens'] == n
            assert final['cached_tokens'] == 0 and final['cached_pages'] == 0
            return {'prefill_tps': 8192 / final['time_prefill'],
                    'decode_tps': (n - 1) / final['time_generate'],
                    'ttft_s': final['time_prefill'], 'new_tokens': n,
                    'cached_tokens': final['cached_tokens'], 'cached_pages': final['cached_pages'],
                    'ids': seq}
        finally:
            gen.close()
            del gen, job
            gc.collect()
            torch.cuda.empty_cache()

    result = {'input_sha256': hashlib.sha256(raw).hexdigest(), 'load_seconds': load_s,
              'protocol': 'one load, default then det; 16-token warm-up per arm; fresh generator per pass',
              'arms': {}}
    for tag in ('default', 'det'):
        if tag == 'det':
            # These are the values used when the three upstream env switches are
            # supplied at process start. Existing host stream states also need
            # their probe-derived threshold changed for this same-load screen.
            os.environ.update(EXL3_MOE_STREAM_T='8', EXL3_MOE_STREAM_FUSED_T='0', EXL3_MOE_RECON_DET='1')
            tuning = moe_cpu_host.TUNING
            tuning.stream_t = 8
            tuning.stream_t_explicit = True
            tuning.stream_fused_t = 0
            moe_batch_recon.DETERMINISTIC = True
            for host in cfg.moe_cpu_hosts.values():
                host.stream_t = 8
                for state in host.sstate.values():
                    state['stream_t'] = 8
                    state['fused_t'] = 0
        run_one(16)
        rows = []
        for rep in range(2):
            row = run_one(256)
            data = struct.pack('<256I', *row.pop('ids'))
            name = f'{tag}-{rep}.bin'
            (out.parent / name).write_bytes(data)
            row.update(rep=rep, ids_file=name, ids_sha256_u32=hashlib.sha256(data).hexdigest())
            rows.append(row)
            print(json.dumps({'event': 'rep', 'arm': tag, **row}), flush=True)
        result['arms'][tag] = {'rows': rows, 'identical': rows[0]['ids_sha256_u32'] == rows[1]['ids_sha256_u32'],
                               'stream_t': moe_cpu_host.TUNING.stream_t,
                               'stream_fused_t': moe_cpu_host.TUNING.stream_fused_t,
                               'recon_deterministic': moe_batch_recon.DETERMINISTIC,
                               'stream_states': [s['stream_t'] for h in cfg.moe_cpu_hosts.values() for s in h.sstate.values()]}
        out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
