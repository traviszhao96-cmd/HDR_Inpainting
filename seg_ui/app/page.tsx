'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Download, Eraser, ImagePlus, KeyRound, Pencil, RotateCcw, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Spinner } from '@/components/ui/spinner';
import { Textarea } from '@/components/ui/textarea';

type Point = { x: number; y: number };
type ModelContext = {
  registerTool: (tool: {
    name: string;
    title: string;
    description: string;
    inputSchema: object;
    annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
    execute: (input: unknown) => unknown;
  }, options?: { signal?: AbortSignal }) => void | Promise<void>;
};

const API_URL = process.env.NEXT_PUBLIC_SEG_API_URL ?? 'http://127.0.0.1:7860';
const SAMPLE_URL = `${API_URL}/sample-uhdr`;

type EraseResult = {
  kind: string;
  preview_url: string;
  download_url: string;
  cloud_input_url: string;
  elapsed_ms: number;
  model: string;
  gainmap_channels: number | null;
  downscale: { source_size: [number, number]; cloud_input_size: [number, number]; cloud_output_size: string };
};

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageBlobRef = useRef<Blob | null>(null);
  const maskBlobRef = useRef<Blob | null>(null);
  const drawingRef = useRef(false);
  const lassoRef = useRef<Point[]>([]);
  const requestRef = useRef(0);
  const tintedMaskRef = useRef<{ url: string; canvas: HTMLCanvasElement } | null>(null);
  const [imageUrl, setImageUrl] = useState(SAMPLE_URL);
  const [fileName, setFileName] = useState('clean_input_uhdr.jpg');
  const [lasso, setLasso] = useState<Point[]>([]);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(48);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('按住画笔，沿着目标外围画一圈');
  const [device, setDevice] = useState('检测中');
  const [serverHasKey, setServerHasKey] = useState(false);
  const [showErase, setShowErase] = useState(false);
  const [erasing, setErasing] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [prompt, setPrompt] = useState('移除蒙版内的整个对象，只重建后方自然背景。保持原有透视、纹理、光线和颜色，不要添加新物体。');
  const [eraseMessage, setEraseMessage] = useState('');
  const [eraseResult, setEraseResult] = useState<EraseResult | null>(null);
  const [contentBackend, setContentBackend] = useState<'opencv' | 'gpt-image-2'>('opencv');

  const loadSampleBlob = useCallback(async () => {
    if (!imageBlobRef.current) imageBlobRef.current = await fetch(SAMPLE_URL).then((response) => response.blob());
  }, []);

  useEffect(() => {
    loadSampleBlob().catch(() => setMessage('示例图片载入失败，请换一张图片'));
    const checkBackend = () => fetch(`${API_URL}/health`).then((response) => response.json()).then((data) => {
      setDevice(data.device ?? '本地');
      setServerHasKey(Boolean(data.openai_configured));
    }).catch(() => setDevice('后端未启动'));
    void checkBackend();
    const timer = window.setInterval(checkBackend, 3000);
    return () => window.clearInterval(timer);
  }, [loadSampleBlob]);

  const clearSelection = useCallback(() => {
    requestRef.current += 1;
    lassoRef.current = [];
    setLasso([]);
    setMaskUrl(null);
    maskBlobRef.current = null;
    tintedMaskRef.current = null;
    setShowErase(false);
    setErasing(false);
    setEraseMessage('');
    setEraseResult(null);
    setLoading(false);
    setMessage('按住画笔，沿着目标外围画一圈');
  }, []);

  useEffect(() => {
    const context = (document as Document & { modelContext?: ModelContext }).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(context.registerTool({
      name: 'clear_object_selection',
      title: '清除对象圈选',
      description: '清除当前画笔圈选和分割蒙版，让用户重新圈选对象。',
      inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute() {
        clearSelection();
        return { cleared: true };
      },
    }, { signal: lifecycle.signal })).catch(() => undefined);
    return () => lifecycle.abort();
  }, [clearSelection]);

  const canvasPoint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.round(((event.clientX - rect.left) / rect.width) * canvas.width),
      y: Math.round(((event.clientY - rect.top) / rect.height) * canvas.height),
    };
  };

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);

    const drawLasso = () => {
      if (lasso.length < 2) return;
      context.beginPath();
      context.moveTo(lasso[0].x, lasso[0].y);
      for (const point of lasso.slice(1)) context.lineTo(point.x, point.y);
      if (!drawingRef.current) context.closePath();
      context.strokeStyle = '#ffd45a';
      context.lineWidth = Math.max(5, canvas.width / 430);
      context.lineCap = 'round';
      context.lineJoin = 'round';
      context.shadowColor = 'rgba(0,0,0,.7)';
      context.shadowBlur = Math.max(4, canvas.width / 700);
      context.stroke();
      context.shadowBlur = 0;
    };

    if (!maskUrl) {
      drawLasso();
      return;
    }

    const drawTintedMask = (tint: HTMLCanvasElement) => {
      context.globalAlpha = opacity / 100;
      context.drawImage(tint, 0, 0);
      context.globalAlpha = 1;
      drawLasso();
    };
    if (tintedMaskRef.current?.url === maskUrl) {
      drawTintedMask(tintedMaskRef.current.canvas);
      return;
    }

    const mask = new Image();
    mask.onload = () => {
      const tint = document.createElement('canvas');
      tint.width = canvas.width;
      tint.height = canvas.height;
      const tintContext = tint.getContext('2d')!;
      tintContext.drawImage(mask, 0, 0, tint.width, tint.height);
      const pixels = tintContext.getImageData(0, 0, tint.width, tint.height);
      for (let index = 0; index < pixels.data.length; index += 4) {
        const maskValue = pixels.data[index];
        pixels.data[index] = 25;
        pixels.data[index + 1] = 230;
        pixels.data[index + 2] = 140;
        pixels.data[index + 3] = maskValue;
      }
      tintContext.putImageData(pixels, 0, 0);
      tintedMaskRef.current = { url: maskUrl, canvas: tint };
      drawTintedMask(tint);
    };
    mask.src = maskUrl;
  }, [lasso, maskUrl, opacity]);

  useEffect(() => drawOverlay(), [drawOverlay]);

  const runSegmentation = async (closedLasso: Point[]) => {
    if (!imageBlobRef.current) await loadSampleBlob();
    if (!imageBlobRef.current || closedLasso.length < 3) return;
    const requestId = ++requestRef.current;
    setLoading(true);
    setMessage('正在识别圈内对象…');
    try {
      const body = new FormData();
      body.append('image', imageBlobRef.current, fileName);
      body.append('lasso', JSON.stringify(closedLasso.map(({ x, y }) => [x, y])));
      body.append('expand', '2');
      const response = await fetch(`${API_URL}/segment`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? '分割失败');
      if (requestId !== requestRef.current) return;
      setMaskUrl(`data:image/png;base64,${data.mask}`);
      const binary = window.atob(data.mask);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      maskBlobRef.current = new Blob([bytes], { type: 'image/png' });
      setDevice(data.device);
      setMessage(`已分出一个对象 · ${data.elapsed_ms} ms`);
    } catch (error) {
      if (requestId === requestRef.current) setMessage(error instanceof Error ? error.message : '无法连接本地模型');
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (loading) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    setMaskUrl(null);
    maskBlobRef.current = null;
    tintedMaskRef.current = null;
    setShowErase(false);
    setEraseResult(null);
    setEraseMessage('');
    const first = canvasPoint(event);
    lassoRef.current = [first];
    setLasso([first]);
    setMessage('继续沿着目标外围画，松开后自动闭合');
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const point = canvasPoint(event);
    const previous = lassoRef.current.at(-1);
    if (previous && Math.hypot(point.x - previous.x, point.y - previous.y) < Math.max(3, canvasRef.current!.width / 900)) return;
    lassoRef.current = [...lassoRef.current, point];
    setLasso(lassoRef.current);
  };

  const finishLasso = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    const closed = lassoRef.current;
    setLasso([...closed]);
    if (closed.length < 6) {
      setMessage('请围绕目标画一个完整的圈');
      return;
    }
    void runSegmentation(closed);
  };

  const onImageLoad = () => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    drawOverlay();
  };

  const chooseImage = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
    imageBlobRef.current = file;
    setImageUrl(URL.createObjectURL(file));
    setFileName(file.name);
    clearSelection();
  };

  const runErase = async () => {
    if (!imageBlobRef.current || !maskBlobRef.current) {
      setEraseMessage('请先完成对象圈选');
      return;
    }
    setErasing(true);
    setEraseResult(null);
    setEraseMessage('正在发送缩小后的局部图，并在本机重建 HDR…');
    try {
      const body = new FormData();
      body.append('image', imageBlobRef.current, fileName);
      body.append('mask', maskBlobRef.current, 'erase_mask.png');
      body.append('prompt', prompt);
      body.append('model', contentBackend);
      body.append('api_key', apiKey);
      const response = await fetch(`${API_URL}/erase`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? '消除失败');
      setEraseResult(data as EraseResult);
      setEraseMessage(`完成 · ${(data.elapsed_ms / 1000).toFixed(1)} 秒`);
    } catch (error) {
      setEraseMessage(error instanceof Error ? error.message : '消除失败');
    } finally {
      setErasing(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#070b0a] text-[#eff8f3]">
      <header className="flex h-16 items-center justify-between border-b border-white/10 px-4 sm:px-7">
        <div className="flex items-center gap-3">
          <div className="grid size-9 place-items-center rounded-xl bg-[#19e68c] text-[#04100a]"><Sparkles className="size-5" /></div>
          <div><h1 className="text-base font-semibold tracking-tight">HDR 对象消除</h1><p className="text-xs text-white/45">本地分割 · 局部云端验证 · 本机重建 HDR</p></div>
        </div>
        <Badge className="border border-[#19e68c]/25 bg-[#19e68c]/10 text-[#83f5bb]"><span className="size-1.5 rounded-full bg-[#19e68c]" />{device}</Badge>
      </header>

      <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-[1680px] flex-col gap-4 p-3 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium"><Pencil className="size-4 text-[#ffd45a]" />圈选画笔</div>
            <p className="mt-1 truncate text-xs text-white/45">{fileName} · {message}</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={clearSelection} disabled={!lasso.length && !maskUrl}><RotateCcw /> 重新圈选</Button>
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={() => fileInputRef.current?.click()}><ImagePlus /> 换图片</Button>
            <input ref={fileInputRef} className="hidden" type="file" accept="image/png,image/jpeg,image/webp,.jpg,.jpeg" onChange={(event) => event.target.files?.[0] && chooseImage(event.target.files[0])} />
          </div>
        </div>

        <div className="relative flex min-h-[55vh] flex-1 items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_50%_30%,#17211d_0%,#070a09_62%)] shadow-2xl">
          <div className="relative max-h-[calc(100vh-13.5rem)] max-w-full select-none">
            <img ref={imageRef} src={imageUrl} alt="待圈选图片" onLoad={onImageLoad} className="block max-h-[calc(100vh-13.5rem)] max-w-full object-contain" draggable={false} />
            <canvas ref={canvasRef} aria-label="用画笔圈选对象" className="absolute inset-0 h-full w-full cursor-crosshair touch-none" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={finishLasso} onPointerCancel={finishLasso} />
          </div>
          {!lasso.length && !loading && <div className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-black/70 px-4 py-2 text-center text-sm text-white/75 shadow-xl backdrop-blur-md">按住鼠标或手指，沿对象外围画一圈</div>}
          {loading && <div className="absolute inset-0 grid place-items-center bg-black/25 backdrop-blur-[1px]"><div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#0b100e]/95 px-4 py-2 text-sm shadow-xl"><Spinner className="text-[#19e68c]" /> 正在分割圈内对象</div></div>}
        </div>

        <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-white/10 bg-white/[0.035] p-3 sm:px-4">
          <div className="min-w-[180px] flex-1 sm:max-w-[290px]"><div className="mb-2 flex justify-between text-xs text-white/55"><span>蒙版透明度</span><span>{opacity}%</span></div><Slider value={[opacity]} min={15} max={80} step={1} onValueChange={(value) => setOpacity(value[0])} /></div>
          <div className="ml-auto flex flex-wrap items-center justify-end gap-2"><span className="hidden text-xs text-white/35 lg:block">黄色：你的圈选　绿色：分割结果</span><a href={maskUrl ?? undefined} download={`${fileName.replace(/\.[^.]+$/, '')}_mask.png`} aria-disabled={!maskUrl} className={`flex h-10 items-center justify-center gap-2 rounded-lg border px-4 text-sm font-medium transition-colors ${maskUrl ? 'border-white/15 bg-white/5 text-white hover:bg-white/10' : 'pointer-events-none border-white/5 text-white/20'}`}><Download className="size-4" /> 只下载蒙版</a><Button className="h-10 bg-[#19e68c] px-4 text-[#04100a] hover:bg-[#6af0ad]" disabled={!maskUrl} onClick={() => setShowErase(true)}><Eraser /> 继续消除</Button></div>
        </div>

        {showErase && !eraseResult && <section className="grid gap-4 rounded-2xl border border-[#19e68c]/20 bg-[#0c1411] p-4 lg:grid-cols-[minmax(0,1fr)_minmax(280px,.7fr)]">
          <div>
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Eraser className="size-4 text-[#19e68c]" />第二步：内容消除</div>
            <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} className="min-h-24 border-white/10 bg-black/20 text-white" aria-label="消除提示词" />
            <p className="mt-2 text-xs leading-5 text-white/45">{contentBackend === 'opencv' ? '本地 OpenCV 消除（免费、无需 Key），用于验证完整 HDR 通路。' : '只上传圈选附近的 SDR 局部，最长边缩至 1024 px；未圈选区域会在本机恢复为原始像素。'}</p>
          </div>
          <div className="flex flex-col justify-between gap-3">
            <div className="space-y-3">
              <div>
                <label className="mb-1.5 flex items-center gap-2 text-sm text-white/70" htmlFor="content-backend"><Eraser className="size-4" />内容引擎</label>
                <select id="content-backend" value={contentBackend} onChange={(event) => setContentBackend(event.target.value as 'opencv' | 'gpt-image-2')} className="h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 text-sm text-white outline-none focus:border-[#19e68c]/60">
                  <option value="opencv" className="bg-[#0c1411]">OpenCV 本地（免费，验证通路）</option>
                  <option value="gpt-image-2" className="bg-[#0c1411]">gpt-image-2 云端（需 API Key，真实消除）</option>
                </select>
              </div>
              {contentBackend !== 'opencv' && <div><label className="mb-2 flex items-center gap-2 text-sm text-white/70" htmlFor="api-key"><KeyRound className="size-4" />OpenAI API Key</label><Input id="api-key" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={serverHasKey ? '后台已设置，可留空' : 'sk-…（仅本次使用，不保存）'} className="h-10 border-white/10 bg-black/20 text-white" autoComplete="off" /></div>}
            </div>
            <div><Button className="h-11 w-full bg-[#19e68c] text-[#04100a] hover:bg-[#6af0ad]" disabled={erasing} onClick={() => void runErase()}>{erasing ? <Spinner /> : <Sparkles />} {erasing ? '正在消除并重建 HDR' : contentBackend === 'opencv' ? '本地 OpenCV 消除并重建 HDR' : '用 gpt-image-2 开始验证'}</Button>{eraseMessage && <p className={`mt-2 text-xs ${eraseMessage.includes('完成') ? 'text-[#75efb1]' : 'text-white/55'}`}>{eraseMessage}</p>}</div>
          </div>
        </section>}

        {eraseResult && <section className="rounded-2xl border border-[#19e68c]/25 bg-[#0c1411] p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2 font-semibold"><CheckCircle2 className="size-5 text-[#19e68c]" />消除完成</div><p className="mt-1 text-xs text-white/45">{eraseResult.kind} · {eraseResult.model} · 输入缩小为 {eraseResult.downscale.cloud_input_size[0]} × {eraseResult.downscale.cloud_input_size[1]}{eraseResult.gainmap_channels ? ' · 单通道 gain map' : ''}</p></div><div className="flex gap-2"><Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={() => { setEraseResult(null); setShowErase(true); }}><RotateCcw /> 再试一次</Button><a className="flex h-10 items-center gap-2 rounded-lg bg-[#19e68c] px-4 text-sm font-medium text-[#04100a] hover:bg-[#6af0ad]" href={`${API_URL}${eraseResult.download_url}`} download><Download className="size-4" /> 下载 {eraseResult.kind}</a></div></div>
          <div className="grid gap-3 md:grid-cols-2"><figure className="overflow-hidden rounded-xl border border-white/10 bg-black/25"><img src={`${API_URL}${eraseResult.cloud_input_url}`} alt="内容引擎输入" className="max-h-80 w-full object-contain" /><figcaption className="border-t border-white/8 px-3 py-2 text-xs text-white/45">{eraseResult.model === 'opencv' ? '本地 OpenCV 输入' : '实际发送给云端的缩小局部'}</figcaption></figure><figure className="overflow-hidden rounded-xl border border-white/10 bg-black/25"><img src={`${API_URL}${eraseResult.preview_url}`} alt="对象消除后的 SDR 预览" className="max-h-80 w-full object-contain" /><figcaption className="border-t border-white/8 px-3 py-2 text-xs text-white/45">本机拼回后的完整 SDR 预览</figcaption></figure></div>
        </section>}
      </section>
    </main>
  );
}
