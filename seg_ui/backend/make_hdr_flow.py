"""Publication-ready monochrome workflow, no synthetic photo content."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out=Path(__file__).resolve().parents[2]/'output/xiaohongshu_hdr_20260917'
im=Image.new('RGB',(1500,2000),'white'); d=ImageDraw.Draw(im)
font='/System/Library/Fonts/STHeiti Medium.ttc'
light='/System/Library/Fonts/STHeiti Light.ttc'
def text(x,y,s,size=32,fill='#181818',regular=False):
    d.text((x,y),s,font=ImageFont.truetype(light if regular else font,size),fill=fill)
def center(y,s,size=32,fill='#181818'):
    f=ImageFont.truetype(font,size); box=d.textbbox((0,0),s,font=f)
    d.text(((1500-(box[2]-box[0]))/2,y),s,font=f,fill=fill)
def arrow(points):
    d.line(points,fill='#666666',width=3)
    x,y=points[-1]; d.polygon([(x,y),(x-9,y-16),(x+9,y-16)],fill='#666666')
def box(rect,title,sub,dark=False):
    x,y,x2,y2=rect
    d.rounded_rectangle(rect,radius=16,fill='#181818' if dark else '#f7f7f7',outline='#181818' if dark else '#d4d4d4',width=2)
    for offset,s,size in [(27,title,38),(89,sub,26)]:
        f=ImageFont.truetype(font if offset==27 else light,size)
        b=d.textbbox((0,0),s,font=f)
        d.text(((x+x2-b[2]+b[0])/2,y+offset),s,font=f,fill=('white' if offset==27 else '#cccccc') if dark else ('#181818' if offset==27 else '#666666'))
text(75,65,'PHOTO LAB  /  HOW IT WORKS',23,'#666666')
d.line((75,120,1425,120),fill='#cccccc',width=2)
text(70,175,'HDR 消除，分两层做',78)
text(75,293,'画面修好了，亮度信息也要一起修。',34,'#666666',True)
box((390,400,1110,550),'原始 Ultra HDR 照片','普通画面 + 亮度增益信息',True)
arrow([(750,550),(750,600),(400,600),(400,670)])
arrow([(750,550),(750,600),(1100,600),(1100,670)])
box((100,680,700,830),'SDR 画面','负责颜色、纹理和内容')
box((800,680,1400,830),'Gainmap 增益图','负责 HDR 亮度增益')
center(886,'同一个目标蒙版，对应两层处理',30)
arrow([(400,830),(400,965)])
arrow([(1100,830),(1100,965)])
box((100,975,700,1135),'SDR 消除','移除目标，重建背景')
box((800,975,1400,1135),'Gainmap 消除','修补目标区域的亮度增益')
arrow([(400,1135),(400,1220),(750,1220),(750,1290)])
arrow([(1100,1135),(1100,1220),(750,1220),(750,1290)])
box((300,1300,1200,1460),'边缘衔接 + HDR 重建','将新画面与新增益结合，重新编码')
arrow([(750,1460),(750,1530)])
box((390,1540,1110,1690),'新的 Ultra HDR 照片','消除路人，保留 HDR 显示能力',True)
d.line((75,1780,1425,1780),fill='#cccccc',width=2)
text(75,1820,'核心：SDR 消除 + Gainmap 消除 + 拼接合成',36)
text(75,1895,'这不是简单叠两张图；还要处理增益、边界和 HDR 编码。',26,'#777777',True)
im.save(out/'workflow.png')
im.save(out/'workflow.jpg',quality=97,subsampling=0)
print(out/'workflow.png')
