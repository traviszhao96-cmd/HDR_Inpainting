"""Compare old/new compositing of the SAME raw generation (no extra GPU call)."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sdr_composite import composite_sdr

parser = argparse.ArgumentParser()
parser.add_argument('results_json', type=Path)
args = parser.parse_args()
row = json.loads(args.results_json.read_text())[0]
root = Path(__file__).resolve().parents[1]
job = root / 'results' / row['result']['job_id']
source = np.asarray(Image.open(job / 'original_sdr.png').convert('RGB'))
mask = np.asarray(Image.open(job / 'mask.png')) > 128
ys, xs = np.nonzero(mask)
padding = max(96, max(xs.max()-xs.min()+1, ys.max()-ys.min()+1))
x0, y0 = max(0, xs.min()-padding), max(0, ys.min()-padding)
x1, y1 = min(source.shape[1], xs.max()+padding+1), min(source.shape[0], ys.max()+padding+1)
src = source[y0:y1, x0:x1]
m = mask[y0:y1, x0:x1]
gen = np.asarray(Image.open(job / 'sdr_generated_crop.png'))
alpha = cv2.GaussianBlur(m.astype(np.uint8)*255, (0,0), 2.5).astype(np.float32)[:,:,None]/255
old = np.clip(src*(1-alpha)+gen*alpha, 0,255).astype(np.uint8)
new = composite_sdr(src, gen, m)
assert np.array_equal(new[~m], src[~m])
box = (max(0,xs.min()-x0-50), max(0,ys.min()-y0-50), min(src.shape[1],xs.max()-x0+51), min(src.shape[0],ys.max()-y0+51))
canvas = Image.new('RGB', (1200, 850), '#202020')
for i, (label, pixels) in enumerate([('Source',src), ('Same generation / old blend',old), ('Same generation / new blend',new)]):
    im = Image.fromarray(pixels).crop(box)
    im.thumbnail((390,800))
    canvas.paste(im,(i*400+(400-im.width)//2,40))
    ImageDraw.Draw(canvas).text((i*400+8,12),label,fill='white')
canvas.save(args.results_json.parent / 'sdr-seam-comparison.png')
print('Outside-mask preservation: PASS')
print(args.results_json.parent / 'sdr-seam-comparison.png')
