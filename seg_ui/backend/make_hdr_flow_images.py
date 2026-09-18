"""3:4 explanatory diagram using the real experiment's images and gain fields."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hdr_ai_erase import load_rgba1010102,compute_luminance_gainmap
out=ROOT/'output/xiaohongshu_hdr_20260917'
before=ROOT/'seg_ui/results/97845d96738b42ec95a7515a9ab04755'
after=ROOT/'seg_ui/results/2d531e0f3741451e96ba1493482609b9'
rgb0=Image.open(before/'original_sdr.png').convert('RGB')
rgb1=Image.open(after/'erased_sdr.png').convert('RGB')
fields=[]
for directory,rgb,raw in [(before,rgb0,'decoded_hdr.raw'),(after,rgb1,'rebuilt_hdr.raw')]:
    hdr=load_rgba1010102(directory/raw,*rgb.size)
    gain,_,_=compute_luminance_gainmap(hdr,np.asarray(rgb,dtype=np.float32))
    fields.append(np.log2(np.maximum(gain,1e-6)))
lo=min(float(np.percentile(v,1)) for v in fields); hi=max(float(np.percentile(v,99)) for v in fields)
gains=[Image.fromarray(np.rint(np.clip((v-lo)/(hi-lo),0,1)*255).astype(np.uint8)).convert('RGB') for v in fields]
im=Image.new('RGB',(1800,2400),'white'); d=ImageDraw.Draw(im)
font='/System/Library/Fonts/STHeiti Medium.ttc'; light='/System/Library/Fonts/STHeiti Light.ttc'
def text(x,y,s,size=30,color='#171717',regular=False):
    d.text((x,y),s,font=ImageFont.truetype(light if regular else font,size),fill=color)
def arrow(points):
    d.line(points,fill='#666666',width=4)
    x,y=points[-1]; d.polygon([(x,y),(x-10,y-18),(x+10,y-18)],fill='#666666')
def card(x,y,title,sub,picture):
    d.rounded_rectangle((x,y,x+780,y+635),radius=16,fill='#f6f6f6',outline='#d8d8d8',width=2)
    text(x+28,y+20,title,37)
    text(x+28,y+77,sub,25,'#666666',True)
    pic=picture.resize((360,480),Image.Resampling.LANCZOS)
    im.paste(pic,(x+210,y+130))
text(80,54,'PHOTO LAB / REAL IMAGE WORKFLOW',25,'#666666')
d.line((80,108,1720,108),fill='#cccccc',width=2)
text(76,148,'路人消除，HDR 也一起修',83)
text(80,262,'原图拆成两层，各自修补，再重新合成。',34,'#666666',True)
card(80,350,'01  原图 / SDR 画面','记录颜色、纹理和画面内容',rgb0)
card(940,350,'01  原始 Gainmap','记录对应位置的亮度增益',gains[0])
arrow([(470,985),(470,1055)])
arrow([(1330,985),(1330,1055)])
card(80,1070,'02  SDR 路人消除','填补背景；下图为实际编辑结果',rgb1)
card(940,1070,'02  Gainmap 区域修补','生成式重建 + 灰度与边缘校正',gains[1])
arrow([(470,1705),(470,1760),(900,1760),(900,1810)])
arrow([(1330,1705),(1330,1760),(900,1760),(900,1810)])
d.rounded_rectangle((80,1830,1720,2250),radius=16,fill='#181818')
im.paste(rgb1.resize((270,360),Image.Resampling.LANCZOS),(115,1860))
text(440,1880,'03  拼接与合成',55,'white')
text(440,1970,'新的 SDR × 对应增益 → HDR 重建',34,'#eeeeee')
text(440,2034,'重新编码为完整 Ultra HDR 照片',33,'#eeeeee')
text(440,2120,'保留 HDR 显示能力，可继续下一轮编辑。',28,'#bbbbbb',True)
text(80,2292,'同一组真实样张 · 完成两轮编辑后的结果',27,'#555555',True)
text(80,2340,'增益图采用统一对数灰度并增强对比，仅作示意；本流程图为 SDR。',24,'#777777',True)
im.save(out/'workflow_with_photos.png')
im.save(out/'workflow_with_photos.jpg',quality=97,subsampling=0)
print(out/'workflow_with_photos.jpg')
