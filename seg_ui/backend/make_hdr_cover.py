"""Deterministic 3:4 HDR editorial cover from actual before/after UHDR files."""
import ctypes as C
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/xiaohongshu_hdr_20260917'
OUT.mkdir(parents=True, exist_ok=True)

class Error(C.Structure):
    _fields_ = [('code',C.c_int),('has_detail',C.c_int),('detail',C.c_char*256)]
class Compressed(C.Structure):
    _fields_ = [('data',C.c_void_p),('size',C.c_size_t),('capacity',C.c_size_t),('cg',C.c_int),('ct',C.c_int),('range',C.c_int)]
class Raw(C.Structure):
    _fields_ = [('fmt',C.c_int),('cg',C.c_int),('ct',C.c_int),('range',C.c_int),('w',C.c_uint),('h',C.c_uint),('planes',C.c_void_p*3),('stride',C.c_uint*3)]
lib = C.CDLL(str(ROOT/'build/libuhdr.1.4.0.dylib'))
lib.uhdr_create_decoder.restype = C.c_void_p
lib.uhdr_release_decoder.argtypes = [C.c_void_p]
lib.uhdr_get_decoded_image.argtypes = [C.c_void_p]
lib.uhdr_get_decoded_image.restype = C.POINTER(Raw)
for name,args in [('uhdr_dec_set_image',[C.c_void_p,C.POINTER(Compressed)]),
                  ('uhdr_dec_set_out_img_format',[C.c_void_p,C.c_int]),
                  ('uhdr_dec_set_out_color_transfer',[C.c_void_p,C.c_int]),
                  ('uhdr_decode',[C.c_void_p])]:
    fn=getattr(lib,name); fn.argtypes=args; fn.restype=Error
def check(status):
    if status.code: raise RuntimeError(status.detail.decode())
def decode(path, hdr):
    data=path.read_bytes(); buffer=C.create_string_buffer(data)
    descriptor=Compressed(C.cast(buffer,C.c_void_p),len(data),len(data),-1,-1,-1)
    ctx=lib.uhdr_create_decoder()
    try:
        check(lib.uhdr_dec_set_image(ctx,C.byref(descriptor)))
        check(lib.uhdr_dec_set_out_img_format(ctx,4 if hdr else 3))
        check(lib.uhdr_dec_set_out_color_transfer(ctx,0 if hdr else 3))
        check(lib.uhdr_decode(ctx))
        r=lib.uhdr_get_decoded_image(ctx).contents
        buf=C.string_at(r.planes[0],r.stride[0]*r.h*(8 if hdr else 4))
        pixels=np.frombuffer(buf,np.float16 if hdr else np.uint8).reshape(r.h,r.stride[0],4)[:,:r.w,:3].astype(np.float32)
        return pixels,r.cg
    finally: lib.uhdr_release_decoder(ctx)
M709=np.array([[.4123908,.35758434,.18048079],[.21263901,.71516868,.07219232],[.01933082,.11919478,.95053215]])
MP3=np.array([[.48657095,.26566769,.19821729],[.22897456,.69173852,.07928691],[0,.04511338,1.04394437]])
M2020=np.array([[.63695805,.14461690,.16888098],[.26270021,.67799807,.05930172],[0,.02807269,1.06098506]])
def linear(v):
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
def srgb(v):
    v=np.maximum(v,0)
    return np.where(v<=.0031308,v*12.92,1.055*v**(1/2.4)-.055)
def to709(v,cg):
    matrix={0:M709,1:MP3,2:M2020}[cg]
    return np.maximum(v @ (np.linalg.inv(M709) @ matrix).T,0).astype(np.float32)

W,H=1800,2400
base=Image.new('RGB',(W,H),'white'); draw=ImageDraw.Draw(base)
font='/System/Library/Fonts/STHeiti Medium.ttc'
regular='/System/Library/Fonts/STHeiti Light.ttc'
latin='/System/Library/Fonts/Supplemental/Arial.ttf'
def text(x,y,s,size=32,color='#181818',face=font):
    draw.text((x,y),s,font=ImageFont.truetype(face,size),fill=color)
text(90,78,'PHOTO LAB    /    EDITING NOTES',25,'#666666',latin)
text(1540,78,'01 / HDR',25,'#666666',latin)
draw.line((90,138,1710,138),fill='#c8c8c8',width=2)
text(82,185,'路人消除',130)
text(82,342,'支持 HDR 啦',130)
text(90,525,'去掉路人，也保留照片里的光。',38,'#626262',regular)
text(90,646,'01   消除前',35)
text(930,646,'02   消除后',35)
boxes=[(90,715,795,1060),(915,715,795,1060)]
paths=[ROOT/'seg_ui/results/97845d96738b42ec95a7515a9ab04755/input.jpeg',
       ROOT/'seg_ui/portfolio/2d531e0f3741451e96ba1493482609b9/work.jpg']
draw.line((90,1850,1710,1850),fill='#c8c8c8',width=2)
text(90,1906,'不止填补画面，也重建 HDR 增益。',49)
text(90,2000,'SDR 内容消除  →  Gainmap 生成式重建  →  UHDR 合成',32,'#555555',regular)
text(90,2073,'FLUX Fill + 消除 LoRA  /  真实样张对比',29,'#666666',regular)
text(90,2230,'HDR 实验记录',28)
text(90,2290,'HDR 效果需兼容的屏幕与查看器；平台转码可能影响显示。',24,'#777777',regular)

# UI white maps to authored 500 nit; photos keep decoded absolute luminance.
sdr=np.asarray(base).copy()
hdr=linear(sdr.astype(np.float32)/255)*(500/203)
sources=[]
for path,(x,y,w,h) in zip(paths,boxes):
    photo,cg=decode(path,False)
    photo_hdr,hcg=decode(path,True)
    photo=srgb(to709(linear(photo/255),cg))
    photo=np.rint(np.clip(photo,0,1)*255).astype(np.uint8)
    photo_hdr=to709(photo_hdr,hcg)
    sdr[y:y+h,x:x+w]=cv2.resize(photo,(w,h),interpolation=cv2.INTER_AREA)
    hdr[y:y+h,x:x+w]=cv2.resize(photo_hdr,(w,h),interpolation=cv2.INTER_AREA)
    sources.append({'path':str(path),'sdr_gamut':cg,'hdr_gamut':hcg})
Image.fromarray(sdr).save(OUT/'cover_sdr_preview.png')
rgba=np.ones((H,W,4),np.float16); rgba[:,:,:3]=np.clip(hdr,0,10000/203)
rgba.tofile(OUT/'cover_linear.raw')
rgba8=np.full((H,W,4),255,np.uint8); rgba8[:,:,:3]=sdr
rgba8.tofile(OUT/'cover_sdr.raw')
subprocess.run([str(ROOT/'build/ultrahdr_app'),'-m','0','-p',str(OUT/'cover_linear.raw'),
    '-y',str(OUT/'cover_sdr.raw'),'-a','4','-b','3','-t','0','-C','0','-c','0',
    '-w',str(W),'-h',str(H),'-q','98','-Q','100','-M','0','-s','1',
    '-z',str(OUT/'cover_hdr.jpg')],check=True,capture_output=True)
decoded,cg=decode(OUT/'cover_hdr.jpg',True)
white_nits=float(np.median(decoded[20:60,20:60])*203)
assert 480<white_nits<520,(white_nits,'white target failed')
assert decoded.shape==(H,W,3) and np.isfinite(decoded).all()
(OUT/'manifest.json').write_text(json.dumps({'size':[W,H],'ratio':'3:4',
 'background_target_nits':500,'decoded_white_nits':white_nits,'sources':sources,
 'note':'Actual display luminance depends on HDR headroom and viewer. PNG preview is SDR.'},indent=2))
print(OUT/'cover_hdr.jpg'); print('Decoded white nits',white_nits)
