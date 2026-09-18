"""Exercise the UI erase API with saved non-church cases and current defaults."""
import json
import argparse
import time
from pathlib import Path
import requests
from PIL import Image, ImageDraw
from sdr_prompt import DEFAULT_SDR_PROMPT

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--only',nargs='+',help='Run only these source filenames')
parser.add_argument('--sdr-steps',type=int,choices=[12,20],default=20)
parser.add_argument('--gainmap-mode',choices=['generative','smooth-interpolation'],default='smooth-interpolation')
args=parser.parse_args()
out=root/'test_runs'/(time.strftime('%Y%m%d-%H%M%S')+'-fill-smooth-pipeline')
out.mkdir(); rows=[]
print('OUTPUT',out,flush=True)
for case_file in sorted((root/'test_masks').glob('*/case.json')):
    case=json.loads(case_file.read_text())
    if case['source_name']=='20260911-104519.jpeg': continue
    if args.only and case['source_name'] not in args.only: continue
    print('START',case['source_name'],flush=True)
    data={'model':'comfyui-flux-fill','prompt':DEFAULT_SDR_PROMPT,'mask_expand':case['mask_expand']}
    data.update(sdr_steps=args.sdr_steps,gainmap_mode=args.gainmap_mode)
    if case.get('test_image'):data['test_image']=case['test_image']
    with (case_file.parent/case['image']).open('rb') as im,(case_file.parent/case['mask']).open('rb') as mask:
        r=requests.post('http://127.0.0.1:7860/erase',data=data,
                        files={'image':(case['source_name'],im),'mask':('mask.png',mask,'image/png')},timeout=360)
    if not r.ok:
        rows.append({'source':case['source_name'],'error':r.text})
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        raise RuntimeError(r.text)
    result=r.json(); job=root/'results'/result['job_id']
    row={'source':case['source_name'],'case':str(case_file),'parameters':data,'result':result}
    panels=[]
    for label,filename in [('Before','original_sdr.png'),('Mask','mask.png'),('After','erased_sdr.png')]:
        im=Image.open(job/filename).convert('RGB');im.thumbnail((400,520))
        panel=Image.new('RGB',(400,550),'#151515');panel.paste(im,((400-im.width)//2,30))
        ImageDraw.Draw(panel).text((8,8),label,fill='white');panels.append(panel)
    montage=Image.new('RGB',(1200,550))
    for i,panel in enumerate(panels):montage.paste(panel,(i*400,0))
    name=Path(case['source_name']).stem+'-comparison.png';montage.save(out/name)
    row['comparison']=str(out/name);rows.append(row)
    (out/'results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    print('DONE',case['source_name'],result['elapsed_ms'],result['kind'],flush=True)
print('COMPLETE',out,flush=True)
