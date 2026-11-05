#!/usr/bin/env python3
"""Generate eight real MaskGen-VQ-L backgrounds, with bounded-memory stages.

The generated backgrounds are NOT labelled traffic-sign images. The separate
copy-paste pipeline places original TRAIN sign cutouts on these backgrounds.
Upstream code: ByteDance 1d-tokenizer, Apache-2.0, pinned in vendor/VENDOR.json.

Run stages in separate processes so only one large model is resident at a time:
  python src/generate_maskgen.py --stage download
  python src/generate_maskgen.py --stage encode --device mps
  python src/generate_maskgen.py --stage generate --device mps
  python src/generate_maskgen.py --stage decode --device mps
Or --stage all runs these stages as sequential subprocesses.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT / '.model-cache'
VENDOR = PROJECT / 'vendor' / '1d-tokenizer'
UPSTREAM_COMMIT = '942a96fbdd873780179d1b78d5462911528bf8c8'
MODELS = {
    'generator': {
        'repo': 'turkeyju/generator_maskgen_vq_l',
        'revision': 'd3fbbe36c3223ab14d6b0019c04eba971a5f8bd4',
        'bytes': 2417292712,
        'sha256': '7021e110ee7d28fe34da72abcce6aa8f882d23f50d978bc7a41b95a2fed568d3',
    },
    'tokenizer': {
        'repo': 'turkeyju/tokenizer_tatitok_bl128_vq',
        'revision': '7650d466a955c754265d20e57513518a4e049f00',
        'bytes': 1564268440,
        'sha256': '4b7b4a6661285d6fb56565061712914a1458d39779f0d5a17f7ed87648f3f196',
    },
}
SCENES = [
    ('night', 'A realistic photograph of an empty residential street at night, warm street lamps, apartment buildings, dark blue sky, realistic asphalt road, no people, no vehicles, no signs, no writing.'),
    ('rain', 'A realistic photograph of an empty residential street in heavy rain at night, wet reflective asphalt, apartment buildings and street lamps, no people, no vehicles, no signs, no writing.'),
    ('dawn', 'A realistic photograph of an empty residential street at dawn, pale sunrise, quiet apartment buildings, realistic asphalt road, no people, no vehicles, no signs, no writing.'),
    ('daylight', 'A realistic photograph of an empty residential street in bright daylight, apartment buildings, green trees, realistic asphalt road, no people, no vehicles, no signs, no writing.'),
    ('overcast', 'A realistic photograph of an empty residential street under an overcast grey sky, apartment buildings, soft flat lighting, asphalt road, no people, no vehicles, no signs, no writing.'),
    ('fog', 'A realistic photograph of an empty residential street in dense morning fog, apartment buildings fading into mist, asphalt road, no people, no vehicles, no signs, no writing.'),
    ('dusk', 'A realistic photograph of an empty residential street at dusk, purple orange sky, apartment buildings and glowing street lamps, asphalt road, no people, no vehicles, no signs, no writing.'),
    ('motion', 'A realistic photograph of an empty residential street seen from a moving camera, slight horizontal motion blur, apartment buildings, asphalt road, no people, no vehicles, no signs, no writing.'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def log(event, **kwargs):
    print(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'event': event, **kwargs}, ensure_ascii=False), flush=True)


def configure_cache(cache):
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['XDG_CACHE_HOME'] = str(cache / 'xdg')
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    sys.path.insert(0, str(VENDOR))


def download_file(url, target, expected_size=None, expected_sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        if expected_sha is None or sha256(target) == expected_sha:
            log('download_cached', file=str(target))
            return
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'traffic-signs-a2-research/1.0',
                                                            **({'Range': f'bytes={offset}-'} if offset else {})})
            with urllib.request.urlopen(request, timeout=120) as response:
                resume = offset > 0 and response.status == 206
                if offset and not resume:
                    offset = 0
                mode = 'ab' if resume else 'wb'
                log('download_start', file=str(target), start_bytes=offset, attempt=attempt+1)
                last_report = time.monotonic()
                with open(partial, mode) as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 15:
                            log('download_progress', file=str(target), bytes=offset, total=expected_size)
                            last_report = time.monotonic()
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(f'Unexpected size: {partial.stat().st_size} != {expected_size}')
            if expected_sha and sha256(partial) != expected_sha:
                partial.unlink()
                raise RuntimeError('SHA256 mismatch; removed invalid partial file')
            partial.replace(target)
            log('download_complete', file=str(target), bytes=target.stat().st_size)
            return
        except Exception as exc:
            log('download_retry', error=str(exc), attempt=attempt+1)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))


def download_models(args):
    for name, spec in MODELS.items():
        directory = args.cache_dir / name
        base = f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}'
        download_file(base + '/config.json', directory / 'config.json')
        download_file(base + '/model.safetensors', directory / 'model.safetensors', spec['bytes'], spec['sha256'])
    (args.cache_dir / 'model_provenance.json').write_text(json.dumps(MODELS, indent=2))



def download_clip_text(args):
    """Retrieve exact text tensors with HTTP Range; omit unused vision tensors."""
    spec = {'repo': 'timm/vit_large_patch14_clip_336.openai',
            'revision': '81e38efc4637de5023b10e75a7f9bd1c6fa6b010',
            'file': 'open_clip_model.safetensors',
            'source_file_sha256': 'fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0'}
    folder = args.cache_dir / 'clip'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'openai_clip_text.safetensors'
    provenance = folder / 'text_provenance.json'
    if target.exists() and provenance.exists():
        if sha256(target) == json.loads(provenance.read_text())['text_file_sha256']:
            log('clip_text_cached', bytes=target.stat().st_size)
            return
    url = f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-262143'}), timeout=120) as r:
        if r.status != 206:
            raise RuntimeError('Server did not honor CLIP header Range request')
        initial = r.read(262144)
    length = struct.unpack('<Q', initial[:8])[0]
    if length + 8 > len(initial):
        raise RuntimeError('Unexpectedly large safetensors header')
    header = json.loads(initial[8:8+length])
    selected = {k:v for k,v in header.items() if k != '__metadata__' and not k.startswith('visual.')}
    end = max(v['data_offsets'][1] for v in selected.values())
    if min(v['data_offsets'][0] for v in selected.values()) != 0:
        raise RuntimeError('Text tensors no longer form the expected leading block')
    if min(v['data_offsets'][0] for k,v in header.items() if k.startswith('visual.')) != end:
        raise RuntimeError('Text tensors are no longer contiguous before vision tensors')
    data_offset = length + 8
    raw = folder / 'clip_text_data.part'
    for attempt in range(6):
        offset = raw.stat().st_size if raw.exists() else 0
        if offset == end:
            break
        try:
            start = data_offset + offset
            last = data_offset + end - 1
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{last}'})
            with urllib.request.urlopen(req, timeout=120) as response:
                expected_range = f'bytes {start}-{last}/'
                if response.status != 206 or not response.headers.get('Content-Range','').startswith(expected_range):
                    raise RuntimeError(f'Unexpected Range response: {response.headers.get("Content-Range")}')
                log('clip_text_download_start', offset=offset, total=end)
                report = time.monotonic()
                with open(raw, 'ab') as out:
                    while block := response.read(1024*1024):
                        out.write(block)
                        offset += len(block)
                        if time.monotonic() - report > 15:
                            log('clip_text_download_progress', bytes=offset, total=end)
                            report = time.monotonic()
            if offset != end:
                raise RuntimeError(f'Incomplete CLIP text data: {offset}/{end}')
            break
        except Exception as exc:
            log('clip_text_download_retry', attempt=attempt+1, error=str(exc))
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 15))
    # Preserve offsets because every selected tensor belongs to the leading block.
    selected['__metadata__'] = {'source': url, 'selection': 'all non-visual tensors, exact unmodified values'}
    encoded_header = json.dumps(selected, separators=(',',':')).encode()
    encoded_header += b' ' * (-len(encoded_header) % 8)
    with open(target, 'wb') as out, open(raw, 'rb') as source:
        out.write(struct.pack('<Q', len(encoded_header)))
        out.write(encoded_header)
        for block in iter(lambda: source.read(8*1024*1024), b''):
            out.write(block)
    raw.unlink()
    spec.update({'text_file_sha256': sha256(target), 'text_bytes': target.stat().st_size,
                 'text_tensor_count': len(selected)-1,
                 'disclosure': 'Exact text-only subset extracted with HTTP Range; full source file was not downloaded or hash-verified. TLS and pinned revision identify source; subset SHA256 recorded.'})
    provenance.write_text(json.dumps(spec, indent=2))
    log('clip_text_download_complete', **spec)

def select_device(args):
    import torch
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; pass --device cpu or use a CUDA/Colab machine.')
    return torch.device(device), torch.float16 if device != 'cpu' else torch.float32


def release():
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_text(args):
    import torch
    import open_clip
    device, dtype = select_device(args)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    log('encode_load_clip', device=str(device))
    from open_clip.transformer import TextTransformer, QuickGELU
    from safetensors.torch import load_file
    download_clip_text(args)
    config = open_clip.get_model_config('ViT-L-14-336')
    with torch.device('meta'):
        clip = TextTransformer(**config['text_cfg'], output_dim=config['embed_dim'], act_layer=QuickGELU)
    state = load_file(str(args.cache_dir / 'clip' / 'openai_clip_text.safetensors'), device='cpu')
    state.pop('logit_scale', None)  # contrastive scale is unused during text encoding
    clip.load_state_dict(state, strict=True, assign=True)
    # Nonpersistent causal-mask buffer was constructed on meta and is absent from weights.
    if clip.attn_mask is not None:
        clip.attn_mask = torch.full((clip.context_length, clip.context_length), float('-inf')).triu_(1)
    del state
    clip.eval().requires_grad_(False).to(device=device, dtype=dtype)
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    # Vendor code uses each residual block with batch-first NLD input directly.
    # open_clip >=2.26 exposes batch_first; match block convention explicitly.
    batch_first = getattr(clip.transformer.resblocks[0].attn, 'batch_first', False)
    encoded = {}
    with torch.inference_mode():
        for text in [''] + [p for _, p in SCENES]:
            ids = tokenizer([text]).to(device)
            x = clip.token_embedding(ids).to(dtype) + clip.positional_embedding.to(dtype)
            if not batch_first:
                x = x.permute(1, 0, 2)
            penultimate = None
            for i, block in enumerate(clip.transformer.resblocks):
                if i == len(clip.transformer.resblocks) - 1:
                    penultimate = x
                x = block(x, attn_mask=clip.attn_mask)
            if not batch_first:
                x = x.permute(1, 0, 2)
                penultimate = penultimate.permute(1, 0, 2)
            full = clip.ln_final(x)
            # Official OpenAI CLIP uses maximum token id (EOT) for pooling.
            # Direct pooling also adapts OpenCLIP3.x's tensor-only return API;
            # upstream MaskGen's legacy helper expects a (pooled,tokens) tuple.
            # The mathematical result is independently verified in generation evidence.
            pooled = full[torch.arange(full.shape[0], device=device), ids.argmax(dim=-1)] @ clip.text_projection
            encoded[text] = {
                'penultimate': penultimate.float().cpu(),
                'pooled': pooled.unsqueeze(1).float().cpu(),
                'text_guidance': full.float().cpu(),
            }
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoded, args.stage_dir / 'text_embeddings.pt')
    meta = {'clip': 'ViT-L-14-336', 'pretrained': 'openai', 'batch_first': batch_first,
            'device': str(device), 'compute_dtype': str(dtype), 'seconds': time.monotonic()-started,
            'prompts': dict(SCENES), 'torch': torch.__version__, 'open_clip': open_clip.__version__,
            'clip_provenance': json.loads((args.cache_dir / 'clip' / 'text_provenance.json').read_text())}
    (args.stage_dir / 'encoding.json').write_text(json.dumps(meta, indent=2))
    log('encode_complete', **meta)


def load_model_low_memory(cls, name, args, device, dtype):
    import torch
    from safetensors.torch import load_file
    from safetensors import safe_open
    directory = args.cache_dir / name
    config = json.loads((directory / 'config.json').read_text())
    # Create no random float32 storage, then assign memory-mapped pretrained tensors.
    # This avoids a second full copy during model loading on an 8GB laptop.
    with torch.device('meta'):
        model = cls(config)
    state = load_file(str(directory / 'model.safetensors'), device='cpu')
    # Hub save_model omits duplicate tied tensors and records alias metadata.
    with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as handle:
        aliases = handle.metadata() or {}
    expected = model.state_dict()
    for alias, canonical in aliases.items():
        if alias in expected and canonical in state and alias not in state:
            state[alias] = state[canonical]
    del expected
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    release()
    return model


def generate_tokens(args):
    import torch
    import modeling.maskgen as upstream
    device, dtype = select_device(args)
    torch.set_num_threads(args.threads)
    text = torch.load(args.stage_dir / 'text_embeddings.pt', map_location='cpu', weights_only=True)
    # Cache exact CLIP outputs instead of keeping CLIP alongside the generator.
    def cached_encoding(_tokenizer, _encoder, captions):
        return (torch.cat([text[c]['penultimate'] for c in captions]).to(device=device, dtype=dtype),
                torch.cat([text[c]['pooled'] for c in captions]).to(device=device, dtype=dtype))
    upstream.open_clip_text_encoding = cached_encoding
    started = time.monotonic()
    log('generator_load', device=str(device), dtype=str(dtype))
    generator = load_model_low_memory(upstream.MaskGen_VQ, 'generator', args, device, dtype)
    tokens = {}
    timings = {}
    with torch.inference_mode(), (torch.autocast(device_type=device.type, dtype=dtype) if device.type != 'cpu' else nullcontext()):
        for i, (condition, prompt) in enumerate(SCENES):
            seed = args.seed + i
            torch.manual_seed(seed)
            if device.type == 'mps':
                torch.mps.manual_seed(seed)
            begin = time.monotonic()
            log('generation_start', condition=condition, seed=seed, prompt=prompt)
            result = generator.generate(captions=[prompt], guidance_scale=12.0,
                                        randomize_temperature=2.0, sample_aesthetic_score=6.5,
                                        num_sample_steps=args.steps, clip_tokenizer=None, clip_encoder=None)
            if device.type == 'mps':
                torch.mps.synchronize()
            tokens[condition] = result.cpu()
            timings[condition] = time.monotonic() - begin
            torch.save(tokens, args.stage_dir / 'generated_tokens.pt')
            log('generation_complete', condition=condition, seconds=timings[condition], tokens=result.shape[-1])
    meta = {'device': str(device), 'compute_dtype': str(dtype), 'steps': args.steps, 'base_seed': args.seed,
            'guidance_scale': 12., 'temperature': 2., 'aesthetic_score': 6.5,
            'seconds_by_condition': timings, 'total_seconds': time.monotonic()-started,
            'torch': torch.__version__, 'diffusers': sys.modules['diffusers'].__version__,
            'open_clip': sys.modules['open_clip'].__version__,
            'models': MODELS, 'vendor_commit': UPSTREAM_COMMIT}
    (args.stage_dir / 'generation.json').write_text(json.dumps(meta, indent=2))
    log('all_tokens_generated', total_seconds=meta['total_seconds'])


def decode_tokens(args):
    import torch
    import numpy as np
    from PIL import Image
    from modeling.tatitok import TATiTok
    device, dtype = select_device(args)
    torch.set_num_threads(args.threads)
    text = torch.load(args.stage_dir / 'text_embeddings.pt', map_location='cpu', weights_only=True)
    tokens = torch.load(args.stage_dir / 'generated_tokens.pt', map_location='cpu', weights_only=True)
    log('tokenizer_load', device=str(device), dtype=str(dtype))
    tokenizer = load_model_low_memory(TATiTok, 'tokenizer', args, device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode(), (torch.autocast(device_type=device.type, dtype=dtype) if device.type != 'cpu' else nullcontext()):
        for i, (condition, prompt) in enumerate(SCENES):
            if condition not in tokens:
                continue
            begin = time.monotonic()
            tensor = tokenizer.decode_tokens(tokens[condition].to(device),
                                               text[prompt]['text_guidance'].to(device=device, dtype=dtype))
            pixels = (tensor.clamp(0, 1) * 255).permute(0, 2, 3, 1).to(device='cpu', dtype=torch.uint8).numpy()[0]
            if not np.isfinite(tensor.float().cpu().numpy()).all():
                raise RuntimeError(f'Non-finite decoded pixels for {condition}')
            name = f'maskgen_{condition}_seed{args.seed+i}.png'
            path = args.output_dir / name
            Image.fromarray(pixels).save(path)
            row = {'file': name, 'condition': condition, 'prompt': prompt, 'seed': args.seed+i,
                   'sha256': sha256(path), 'width': int(pixels.shape[1]), 'height': int(pixels.shape[0]),
                   'synthetic': True, 'generator': 'MaskGen-VQ-L', 'tokenizer': 'TA-TiTok-BL128-VQ',
                   'purpose': 'Synthetic background for class-preserving TRAIN-only traffic-sign copy-paste.',
                   'decode_seconds': time.monotonic()-begin}
            rows.append(row)
            log('background_saved', **row)
    provenance = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'vendor_commit': UPSTREAM_COMMIT, 'models': MODELS,
                  'generation': json.loads((args.stage_dir / 'generation.json').read_text()),
                  'encoding': json.loads((args.stage_dir / 'encoding.json').read_text()),
                  'backgrounds': rows,
                  'disclosure': 'All listed backgrounds were synthesized using actual pretrained MaskGen-VQ-L '
                                'and TA-TiTok. Final traffic-sign images are hybrid composites with real TRAIN '
                                'cutouts. Prompts request empty scenes; generated content still requires visual QA.'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['download', 'download-clip', 'encode', 'generate', 'decode', 'all'], default='all')
    p.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--output-dir', type=Path, default=PROJECT / 'data' / 'maskgen_backgrounds')
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=4200)
    p.add_argument('--steps', type=int, default=16)
    args = p.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.stage_dir = args.cache_dir / 'maskgen_stage'
    configure_cache(args.cache_dir)
    if args.stage == 'all':
        for stage in ['download','encode','generate','decode']:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                            '--cache-dir', str(args.cache_dir), '--output-dir', str(args.output_dir),
                            '--device', args.device, '--threads', str(args.threads), '--seed', str(args.seed),
                            '--steps', str(args.steps)], check=True)
    else:
        {'download': download_models, 'download-clip': download_clip_text, 'encode': encode_text, 'generate': generate_tokens, 'decode': decode_tokens}[args.stage](args)

if __name__ == '__main__':
    main()
