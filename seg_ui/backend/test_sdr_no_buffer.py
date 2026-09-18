"""Recover prior seed by matching saved raw pixels, then test no buffer."""
import io
import json
import time
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw
from .app import comfyui_downscaled_edit, COMFYUI_URL

root = Path(__file__).resolve().parents[1]
old = root / 'results/32c8d7df178140568fc558e6eb145145'
raw = np.asarray(Image.open(old / 'sdr_generated_crop.png'))
session = requests.Session()
session.trust_env = False
response = session.get(COMFYUI_URL + '/history', timeout=30)
response.raise_for_status()
match = None
for record in reversed(list(response.json().values())):
    graph = record['prompt'][2]
    if graph.get('4', {}).get('inputs', {}).get('unet_name') != 'flux1-fill-dev-Q4_K_S.gguf':
        continue
    for info in record.get('outputs', {}).get('19', {}).get('images', []):
        res = session.get(COMFYUI_URL + '/view', params=info, timeout=30)
        res.raise_for_status()
        pixels = np.asarray(Image.open(io.BytesIO(res.content)).convert('RGB').resize(
            (raw.shape[1], raw.shape[0]), Image.Resampling.LANCZOS))
        if np.array_equal(raw, pixels):
            match = graph
            break
    if match is not None:
        break
if match is None:
    raise RuntimeError('Cannot verify previous seed; do not run an uncontrolled comparison')
seed = match['17']['inputs']['seed']
print('VERIFIED prior raw output; seed', seed, flush=True)
out = root / 'test_runs' / (time.strftime('%Y%m%d-%H%M%S') + '-sdr-no-buffer')
out.mkdir()
source = np.asarray(Image.open(old / 'original_sdr.png').convert('RGB'))
mask = np.asarray(Image.open(old / 'mask.png')) > 128
start = time.monotonic()
new, info = comfyui_downscaled_edit(source, mask, prompt=match['6']['inputs']['text'],
    cloud_input_path=out/'cloud_input.png', engine_override='flux-fill', seed=seed)
Image.fromarray(new).save(out/'erased_sdr.png')
assert np.array_equal(new[~mask], source[~mask])
assert np.array_equal(np.asarray(Image.open(out/'sdr_generation_mask.png')),
                      np.asarray(Image.open(out/'sdr_composite_mask.png')))
ys, xs = np.nonzero(mask)
box = (max(0,xs.min()-60),max(0,ys.min()-60),min(source.shape[1],xs.max()+61),min(source.shape[0],ys.max()+61))
canvas = Image.new('RGB',(1200,850),'#202020')
for i,(label,im) in enumerate([
    ('Original',Image.fromarray(source)),
    ('10px buffer / same seed',Image.open(old/'erased_sdr.png')),
    ('No buffer / same seed',Image.fromarray(new))]):
    im = im.crop(box); im.thumbnail((390,800))
    canvas.paste(im,(i*400+(400-im.width)//2,40))
    ImageDraw.Draw(canvas).text((i*400+8,12),label,fill='white')
canvas.save(out/'comparison.png')
(out/'report.json').write_text(json.dumps({'seed':seed,'previous':str(old),
    'elapsed_s':time.monotonic()-start,'info':info,'outside_unchanged':True,
    'generation_mask_equals_composite_mask':True},indent=2))
print('COMPLETE',out,flush=True)
