'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Download, Eraser, ImagePlus, Paintbrush, Pencil, RotateCcw, Sparkles, Undo2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Spinner } from '@/components/ui/spinner';
import { Textarea } from '@/components/ui/textarea';

type Point = { x: number; y: number };
type TestImage = { name: string; url: string };
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
const DEFAULT_TEST_IMAGE = 'clean_input_uhdr.jpg';
const SAMPLE_URL = `${API_URL}/test-images/${DEFAULT_TEST_IMAGE}/preview`;
const sliderValue = (value: number | readonly number[]) => typeof value === 'number' ? value : value[0];

type EraseResult = {
  job_id: string;
  comparisons?: { sdr: [string, string]; gainmap: [string, string] | null; hdr: [string, string] | null };
  kind: string;
  preview_url: string;
  download_url: string;
  cloud_input_url: string;
  elapsed_ms: number;
  model: string;
  gainmap_channels: number | null;
  timings_seconds?: { preparation: number; sdr: number; gainmap_and_encode: number; total: number };
  downscale: { source_size: [number, number]; cloud_input_size: [number, number]; cloud_output_size: string };
};
type PortfolioWork = { id: string; name: string; kind: string; url: string; preview_url: string; path: string };

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageBlobRef = useRef<Blob | null>(null);
  const sampleLoadRef = useRef<Promise<void> | null>(null);
  const maskBlobRef = useRef<Blob | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskHistoryRef = useRef<Blob[]>([]);
  const drawingRef = useRef(false);
  const lassoRef = useRef<Point[]>([]);
  const requestRef = useRef(0);
  const tintedMaskRef = useRef<{ source: HTMLCanvasElement; version: number; canvas: HTMLCanvasElement } | null>(null);
  const [imageUrl, setImageUrl] = useState('');
  const [fileName, setFileName] = useState('clean_input_uhdr.jpg');
  const [lasso, setLasso] = useState<Point[]>([]);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [maskVersion, setMaskVersion] = useState(0);
  const [editMode, setEditMode] = useState<'lasso' | 'add' | 'erase'>('lasso');
  const [brushSize, setBrushSize] = useState(48);
  const [maskInward, setMaskInward] = useState(0);
  const [maskOutward, setMaskOutward] = useState(2);
  const [maskSmooth, setMaskSmooth] = useState(3);
  const [savingMask, setSavingMask] = useState(false);
  const [maskSaveMessage, setMaskSaveMessage] = useState('');
  const [maskExpand, setMaskExpand] = useState(8);
  const [canUndoMask, setCanUndoMask] = useState(false);
  const [opacity, setOpacity] = useState(48);
  const [searchMargin, setSearchMargin] = useState(48);
  const [imageReady, setImageReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('按住画笔，沿着目标外围画一圈');
  const [device, setDevice] = useState('检测中');
  const [erasing, setErasing] = useState(false);
  const [prompt, setPrompt] = useState('Remove all selected subjects and objects inside the mask. Reconstruct only empty, unoccupied background surfaces, continuing the surrounding geometry, textures and color gradients. Match perspective, lighting and colors. Do not generate any people, faces, heads, bodies, limbs, human silhouettes or portraits inside the mask. Do not replace the removed subject with another person or extend nearby people into the mask. Preserve existing people outside the mask. Do not add new objects.');
  const [eraseMessage, setEraseMessage] = useState('');
  const [eraseProgress, setEraseProgress] = useState(0);
  const [eraseElapsed, setEraseElapsed] = useState(0);
  const [eraseStage, setEraseStage] = useState('正在上传图片与蒙版…');
  const [eraseResult, setEraseResult] = useState<EraseResult | null>(null);
  const [currentWork, setCurrentWork] = useState<EraseResult | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioWork[]>([]);
  const [workMessage, setWorkMessage] = useState('');
  const [savingWork, setSavingWork] = useState(false);
  useEffect(() => {
    fetch(`${API_URL}/portfolio`).then(r => {
      if (!r.ok) throw new Error('作品集加载失败');
      return r.json() as Promise<{items: PortfolioWork[]}>;
    }).then(data => setPortfolio(data.items)).catch(() => setWorkMessage('作品集暂不可用，请确认后端已启动'));
  }, []);
  const [contentBackend, setContentBackend] = useState<'opencv' | 'comfyui-flux-fill'>('comfyui-flux-fill');
  const [gainmapMode, setGainmapMode] = useState('generative');
  const [sdrSteps, setSdrSteps] = useState(20);
  const [testImages, setTestImages] = useState<TestImage[]>([]);
  const [testImageIndex, setTestImageIndex] = useState(-1);
  const [testSourceName, setTestSourceName] = useState<string | null>(DEFAULT_TEST_IMAGE);

  useEffect(() => {
    if (!erasing) return;
    const started = performance.now();
    const timer = window.setInterval(() => {
      const seconds = (performance.now() - started) / 1000;
      setEraseElapsed(Math.floor(seconds));
      // Estimates only: no server milestones. Generative HDR budget ~120s.
      const phases = gainmapMode === 'generative'
        ? [{end:30, percent:20, label:'准备并上传图片与蒙版（预计）…'},
           {end:75, percent:60, label:'生成 SDR 消除内容并回传（预计）…'},
           {end:110, percent:90, label:'生成 Gainmap 并校正边缘（预计）…'},
           {end:120, percent:95, label:'合成 HDR 并载入新底图（预计）…'}]
        : contentBackend === 'opencv'
          ? [{end:3, percent:20, label:'准备图片与蒙版（预计）…'},
             {end:10, percent:95, label:'本地重建与合成（预计）…'}]
          : [{end:30, percent:25, label:'准备并上传图片与蒙版（预计）…'},
             {end:80, percent:85, label:'生成 SDR 消除内容并回传（预计）…'},
             {end:90, percent:95, label:'Gainmap 插值与 HDR 合成（预计）…'}];
      let previousEnd = 0;
      let previousPercent = 0;
      for (const phase of phases) {
        if (seconds < phase.end) {
          setEraseProgress(Math.floor(previousPercent + (phase.percent - previousPercent) *
            (seconds - previousEnd) / (phase.end - previousEnd)));
          setEraseStage(phase.label);
          return;
        }
        previousEnd = phase.end;
        previousPercent = phase.percent;
      }
      setEraseProgress(Math.min(99, Math.floor(95 + 4 * (1 - Math.exp(-(seconds - previousEnd) / 60)))));
      setEraseStage('已超过预计时长，仍在等待处理结果，请勿重复提交…');
    }, 250);
    return () => window.clearInterval(timer);
  }, [erasing, gainmapMode, contentBackend]);

  const loadSampleBlob = useCallback(async () => {
    if (imageBlobRef.current) return;
    if (!sampleLoadRef.current) {
      setImageReady(false);
      sampleLoadRef.current = (async () => {
        const response = await fetch(SAMPLE_URL);
        if (!response.ok) throw new Error(`示例图片载入失败（HTTP ${response.status}）`);
        const blob = await response.blob();
        imageBlobRef.current = blob;
        setImageUrl(URL.createObjectURL(blob));
        setImageReady(true);
      })().finally(() => {
        sampleLoadRef.current = null;
      });
    }
    await sampleLoadRef.current;
  }, []);

  useEffect(() => {
    loadSampleBlob().catch(() => {
      setImageReady(false);
      setMessage('示例图片载入失败，请确认后端已启动，或换一张图片');
    });
    fetch(`${API_URL}/test-images`).then((response) => response.json()).then((data) => {
      setTestImages(Array.isArray(data.items) ? data.items : []);
      if (Array.isArray(data.items) && data.items.length) setTestImageIndex(0);
    }).catch(() => setTestImages([]));
    const checkBackend = () => fetch(`${API_URL}/health`).then((response) => response.json()).then((data) => {
      setDevice(data.device ?? '本地');
      if (!imageBlobRef.current) {
        void loadSampleBlob()
          .then(() => setMessage('按住画笔，沿着目标外围画一圈'))
          .catch(() => setMessage('示例图片载入失败，请确认后端已启动，或换一张图片'));
      }
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
    maskCanvasRef.current = null;
    maskHistoryRef.current = [];
    setMaskVersion((value) => value + 1);
    setCanUndoMask(false);
    setEditMode('lasso');
    tintedMaskRef.current = null;
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
      if (!drawingRef.current && lasso.length > 2) {
        const xs = lasso.map((point) => point.x);
        const ys = lasso.map((point) => point.y);
        const left = Math.max(0, Math.min(...xs) - searchMargin);
        const top = Math.max(0, Math.min(...ys) - searchMargin);
        const right = Math.min(canvas.width, Math.max(...xs) + searchMargin);
        const bottom = Math.min(canvas.height, Math.max(...ys) + searchMargin);
        context.save();
        context.strokeStyle = 'rgba(255, 255, 255, .65)';
        context.lineWidth = Math.max(3, canvas.width / 700);
        context.setLineDash([Math.max(10, canvas.width / 250), Math.max(7, canvas.width / 360)]);
        context.strokeRect(left, top, right - left, bottom - top);
        context.restore();
      }
      context.beginPath();
      context.moveTo(lasso[0].x, lasso[0].y);
      for (const point of lasso.slice(1)) context.lineTo(point.x, point.y);
      if (!drawingRef.current) context.closePath();
      context.strokeStyle = '#ffffff';
      context.lineWidth = Math.max(5, canvas.width / 430);
      context.lineCap = 'round';
      context.lineJoin = 'round';
      context.shadowColor = 'rgba(0,0,0,.7)';
      context.shadowBlur = Math.max(4, canvas.width / 700);
      context.stroke();
      context.shadowBlur = 0;
    };

    const maskCanvas = maskCanvasRef.current;
    if (!maskUrl || !maskCanvas) {
      drawLasso();
      return;
    }

    // The editable PNG is opaque black/white. source-in uses alpha, so it
    // would color the entire image, including the unselected black pixels.
    let tint = tintedMaskRef.current;
    if (!tint || tint.source !== maskCanvas || tint.version !== maskVersion) {
      const tintCanvas = document.createElement('canvas');
      tintCanvas.width = maskCanvas.width;
      tintCanvas.height = maskCanvas.height;
      const tintContext = tintCanvas.getContext('2d')!;
      const pixels = maskCanvas.getContext('2d')!.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      for (let index = 0; index < pixels.data.length; index += 4) {
        const coverage = Math.round(pixels.data[index] * pixels.data[index + 3] / 255);
        pixels.data[index] = 238;
        pixels.data[index + 1] = 238;
        pixels.data[index + 2] = 238;
        pixels.data[index + 3] = coverage;
      }
      tintContext.putImageData(pixels, 0, 0);
      tint = { source: maskCanvas, version: maskVersion, canvas: tintCanvas };
      tintedMaskRef.current = tint;
    }
    context.save();
    context.globalAlpha = opacity / 100;
    context.drawImage(tint.canvas, 0, 0, canvas.width, canvas.height);
    context.restore();
    drawLasso();
  }, [lasso, maskUrl, maskVersion, opacity, searchMargin]);

  useEffect(() => drawOverlay(), [drawOverlay]);

  const runSegmentation = async (closedLasso: Point[]) => {
    const requestId = ++requestRef.current;
    setLoading(true);
    setMessage('正在识别圈内对象…');
    try {
      if (!imageBlobRef.current) await loadSampleBlob();
      const sourceBlob = imageBlobRef.current;
      if (!sourceBlob) throw new Error('图片尚未载入，请稍后重试');
      if (closedLasso.length < 3) throw new Error('请完整地圈住一个对象');
      const body = new FormData();
      body.append('image', sourceBlob, fileName);
      if (testSourceName) body.append('test_image', testSourceName);
      body.append('lasso', JSON.stringify(closedLasso.map(({ x, y }) => [x, y])));
      body.append('expand', '2');
      body.append('search_margin', String(searchMargin));
      const response = await fetch(`${API_URL}/segment`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? '分割失败');
      if (requestId !== requestRef.current) return;
      const dataUrl = `data:image/png;base64,${data.mask}`;
      setMaskUrl(dataUrl);
      const binary = window.atob(data.mask);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      const blob = new Blob([bytes], { type: 'image/png' });
      maskBlobRef.current = blob;
      const maskImage = new Image();
      maskImage.onload = () => {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = maskImage.naturalWidth;
        maskCanvas.height = maskImage.naturalHeight;
        maskCanvas.getContext('2d')!.drawImage(maskImage, 0, 0);
        maskCanvasRef.current = maskCanvas;
        maskHistoryRef.current = [];
        setCanUndoMask(false);
        setEditMode('add');
        setMaskVersion((value) => value + 1);
      };
      maskImage.src = dataUrl;
      setDevice(data.device);
      setMessage(`已生成初始蒙版 · 请用添加/擦除画笔检查细节 · ${data.elapsed_ms} ms`);
    } catch (error) {
      if (requestId === requestRef.current) setMessage(error instanceof Error ? error.message : '无法连接本地模型');
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (loading) return;
    if (!imageReady) {
      setMessage('图片正在载入，请稍后再圈选');
      return;
    }
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drawingRef.current = true;
    if (maskUrl && maskCanvasRef.current && editMode !== 'lasso') {
      const snapshotCanvas = maskCanvasRef.current;
      snapshotCanvas.toBlob((blob) => {
        if (!blob) return;
        maskHistoryRef.current = [...maskHistoryRef.current.slice(-7), blob];
        setCanUndoMask(true);
      }, 'image/png');
      const first = canvasPoint(event);
      lassoRef.current = [first];
      const maskContext = snapshotCanvas.getContext('2d')!;
      maskContext.fillStyle = editMode === 'add' ? '#fff' : '#000';
      maskContext.beginPath();
      maskContext.arc(first.x, first.y, brushSize / 2, 0, Math.PI * 2);
      maskContext.fill();
      setMaskVersion((value) => value + 1);
      setMessage(editMode === 'add' ? '正在补入漏选区域' : '正在擦除误选区域');
      return;
    }
    setMaskUrl(null);
    maskBlobRef.current = null;
    tintedMaskRef.current = null;
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
    if (maskUrl && maskCanvasRef.current && editMode !== 'lasso' && previous) {
      const maskContext = maskCanvasRef.current.getContext('2d')!;
      maskContext.strokeStyle = editMode === 'add' ? '#fff' : '#000';
      maskContext.lineWidth = brushSize;
      maskContext.lineCap = 'round';
      maskContext.lineJoin = 'round';
      maskContext.beginPath();
      maskContext.moveTo(previous.x, previous.y);
      maskContext.lineTo(point.x, point.y);
      maskContext.stroke();
      lassoRef.current = [...lassoRef.current, point];
      setMaskVersion((value) => value + 1);
      return;
    }
    lassoRef.current = [...lassoRef.current, point];
    setLasso(lassoRef.current);
  };

  const finishLasso = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (maskUrl && maskCanvasRef.current && editMode !== 'lasso') {
      maskCanvasRef.current.toBlob((blob) => {
        if (!blob) return;
        maskBlobRef.current = blob;
        setMaskUrl((previous) => {
          if (previous?.startsWith('blob:')) URL.revokeObjectURL(previous);
          return URL.createObjectURL(blob);
        });
        setMaskVersion((value) => value + 1);
      }, 'image/png');
      setMessage('蒙版已修改，可以继续修补或进入消除');
      return;
    }
    if (lassoRef.current.length < 6) {
      setMessage('请围绕目标画一个完整的圈');
      return;
    }
    const first = lassoRef.current[0];
    const last = lassoRef.current.at(-1)!;
    const closed = Math.hypot(first.x - last.x, first.y - last.y) > 1
      ? [...lassoRef.current, { ...first }]
      : [...lassoRef.current];
    lassoRef.current = closed;
    setLasso(closed);
    setMessage(`套索已闭合，正在扩大 ${searchMargin}px 搜索对象…`);
    void runSegmentation(closed);
  };

  const saveTestMask = async () => {
    if (!maskBlobRef.current || !imageBlobRef.current || savingMask || loading || erasing) return;
    setSavingMask(true);
    setMaskSaveMessage('正在保存到本地…');
    try {
      const body = new FormData();
      body.append('image', imageBlobRef.current, fileName);
      body.append('mask', maskBlobRef.current, 'mask.png');
      if (testSourceName) body.append('test_image', testSourceName);
      body.append('prompt', prompt);
      body.append('model', contentBackend);
      body.append('mask_expand', String(maskExpand));
      const response = await fetch(`${API_URL}/test-masks/save`, {method: 'POST', body});
      const data = await response.json() as {path?: string; detail?: string};
      if (!response.ok || !data.path) throw new Error(data.detail || '保存失败');
      setMaskSaveMessage(`已保存：${data.path}`);
    } catch (error) {
      setMaskSaveMessage(error instanceof Error ? error.message : '保存失败，请检查本地后端');
    } finally { setSavingMask(false); }
  };

  const cleanCurrentMask = async () => {
    if (!maskBlobRef.current || loading || erasing) return;
    const previous = maskBlobRef.current;
    const requestId = ++requestRef.current;
    setLoading(true);
    try {
      const body = new FormData();
      body.append('mask', previous, 'mask.png');
      body.append('inward', String(maskInward));
      body.append('outward', String(maskOutward));
      body.append('smooth', String(maskSmooth));
      const response = await fetch(`${API_URL}/mask/refine`, {method: 'POST', body});
      if (!response.ok) throw new Error('蒙版优化失败，请检查收缩参数及后端连接');
      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob);
      if (requestId !== requestRef.current) { bitmap.close(); return; }
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
      bitmap.close();
      maskHistoryRef.current.push(previous);
      maskCanvasRef.current = canvas;
      maskBlobRef.current = blob;
      setMaskUrl((old) => { if (old?.startsWith('blob:')) URL.revokeObjectURL(old); return URL.createObjectURL(blob); });
      setCanUndoMask(true);
      setMaskVersion(v => v + 1);
      setEraseResult(null);
      setMessage('已填孔并平滑边缘，应用向内收缩／向外扩展；可撤销');
    } catch (error) {
      if (requestId === requestRef.current) setMessage(error instanceof Error ? error.message : '蒙版优化失败');
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  };

  const useLassoAsMask = () => {
    const canvas = canvasRef.current;
    if (!canvas || lasso.length < 3) return;
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const context = maskCanvas.getContext('2d')!;
    context.fillStyle = '#000';
    context.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    context.beginPath();
    context.moveTo(lasso[0].x, lasso[0].y);
    for (const point of lasso.slice(1)) context.lineTo(point.x, point.y);
    context.closePath();
    context.fillStyle = '#fff';
    context.fill();
    maskCanvasRef.current = maskCanvas;
    maskHistoryRef.current = [];
    setCanUndoMask(false);
    setEditMode('add');
    maskCanvas.toBlob((blob) => {
      if (!blob) return;
      maskBlobRef.current = blob;
      setMaskUrl(URL.createObjectURL(blob));
      setMaskVersion((value) => value + 1);
      setMessage('已改用闭合套索作为蒙版，请用画笔修补边缘');
    }, 'image/png');
  };

  const undoMaskEdit = () => {
    const blob = maskHistoryRef.current.pop();
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const context = maskCanvas.getContext('2d')!;
      context.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      context.drawImage(image, 0, 0);
      maskBlobRef.current = blob;
      setMaskUrl((previous) => {
        if (previous?.startsWith('blob:')) URL.revokeObjectURL(previous);
        return URL.createObjectURL(blob);
      });
      setCanUndoMask(maskHistoryRef.current.length > 0);
      setMaskVersion((value) => value + 1);
      setMessage('已撤销上一次蒙版修改');
      URL.revokeObjectURL(url);
    };
    image.src = url;
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
    if (erasing || loading) return;
    if (!file.type.startsWith('image/')) return;
    if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
    imageBlobRef.current = file;
    setImageReady(true);
    setImageUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setTestImageIndex(-1);
    setTestSourceName(null);
    clearSelection();
    setCurrentWork(null);
    setWorkMessage('');
  };

  const chooseTestImage = async (index: number) => {
    if (erasing || loading) return;
    const item = testImages[index];
    if (!item) return;
    setImageReady(false);
    try {
      const response = await fetch(`${API_URL}${item.url}`);
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
      imageBlobRef.current = blob;
      setImageReady(true);
      setImageUrl(URL.createObjectURL(blob));
      setFileName(item.name);
      setTestImageIndex(index);
      setTestSourceName(item.name);
      clearSelection();
      setCurrentWork(null);
      setWorkMessage('');
    } catch {
      setImageReady(false);
      setMessage('测试图片载入失败');
    }
  };

  const stepTestImage = (offset: number) => {
    if (!testImages.length) return;
    const current = testImageIndex < 0 ? 0 : testImageIndex;
    void chooseTestImage((current + offset + testImages.length) % testImages.length);
  };

  const runErase = async () => {
    if (erasing) return;
    if (!imageBlobRef.current || !maskBlobRef.current) {
      setEraseMessage('请先完成对象圈选');
      return;
    }
    setErasing(true);
    setEraseProgress(0);
    setEraseElapsed(0);
    setEraseStage('正在上传图片与蒙版…');
    setEraseResult(null);
    setEraseMessage('');
    try {
      const body = new FormData();
      body.append('image', imageBlobRef.current, fileName);
      if (testSourceName) body.append('test_image', testSourceName);
      body.append('mask', maskBlobRef.current, 'erase_mask.png');
      body.append('gainmap_mode', gainmapMode);
      body.append('sdr_steps', String(sdrSteps));
      body.append('prompt', prompt);
      body.append('model', contentBackend);
      body.append('mask_expand', String(maskExpand));
      const response = await fetch(`${API_URL}/erase`, { method: 'POST', body });
      const data = await response.json() as EraseResult & {detail?: string};
      if (!response.ok) throw new Error(data.detail ?? '消除失败');
      setCurrentWork(data);
      // Next edit uploads the actual UHDR, never the SDR comparison preview.
      const nextResponse = await fetch(`${API_URL}${data.download_url}`);
      if (!nextResponse.ok) throw new Error('消除已完成，但新底图加载失败；仍可保存到作品集');
      const nextBlob = await nextResponse.blob();
      imageBlobRef.current = nextBlob;
      if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
      setImageUrl(`${API_URL}${data.preview_url}`);
      setFileName(`edited-${data.job_id}.${data.kind === 'Ultra HDR' ? 'jpg' : 'png'}`);
      setTestSourceName(null);
      setTestImageIndex(-1);
      clearSelection();
      setImageReady(true);
      setEraseResult(data);
      setWorkMessage('新图已成为当前底图，可继续圈选消除，或保存到作品集');
      setEraseProgress(100);
      setEraseMessage(`消除与合成完成 · 100% · ${(data.elapsed_ms / 1000).toFixed(1)} 秒`);
    } catch (error) {
      setEraseMessage(error instanceof Error ? error.message : '消除失败');
    } finally {
      setErasing(false);
    }
  };

  const saveWork = async () => {
    if (!currentWork || savingWork) return;
    setSavingWork(true);
    try {
      const body = new FormData(); body.append('job_id', currentWork.job_id);
      const response = await fetch(`${API_URL}/portfolio`, {method:'POST', body});
      if (!response.ok) throw new Error('作品保存失败');
      const item = await response.json() as PortfolioWork;
      setPortfolio(previous => [item, ...previous.filter(work => work.id !== item.id)]);
      setWorkMessage(`已保存到本地：${item.path}`);
    } catch (error) { setWorkMessage(error instanceof Error ? error.message : '保存失败'); }
    finally { setSavingWork(false); }
  };

  const openWork = async (work: PortfolioWork) => {
    if (erasing || loading) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}${work.url}`);
      if (!response.ok) throw new Error('无法打开作品');
      const blob = await response.blob();
      if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
      imageBlobRef.current = blob;
      setImageUrl(`${API_URL}${work.preview_url}`);
      setFileName(work.url.split('/').at(-1)!);
      setTestSourceName(null); setTestImageIndex(-1);
      clearSelection(); setCurrentWork(null); setImageReady(true);
      setWorkMessage('作品已载入，可继续编辑；原作品不会被覆盖');
    } catch (error) { setWorkMessage(error instanceof Error ? error.message : '加载失败'); }
    finally { setLoading(false); }
  };

  return (
    <main className="min-h-screen bg-[#101010] text-[#f2f2f2]">
      <header className="flex h-16 items-center justify-between border-b border-white/10 px-4 sm:px-7">
        <div className="flex items-center gap-3">
          <div className="grid size-9 place-items-center rounded-xl bg-[#ededed] text-[#151515]"><Sparkles className="size-5" /></div>
          <div><p className="mb-1 text-[10px] font-medium tracking-[0.28em] text-white/45">HDR / PHOTO STUDIO</p><h1 className="text-base font-semibold tracking-tight">HDR 对象消除</h1><p className="text-xs text-white/45">本地圈选 · SDR 消除 · 可选生成式 Gainmap · HDR 合成</p></div>
        </div>
        <Badge className="border border-[#ededed]/25 bg-[#ededed]/10 text-[#dddddd]"><span className="size-1.5 rounded-full bg-[#ededed]" />{device}</Badge>
      </header>

      <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-[1680px] flex-col gap-4 p-3 sm:p-5">
        <ol aria-label="处理流程" className="grid grid-cols-2 gap-2 rounded-xl border border-white/10 bg-[#181818] p-3 text-xs sm:grid-cols-4">
          {['1 · 圈选并确认蒙版', '2 · SDR 内容消除', '3 · Gainmap 重建', '4 · 合成并对比 HDR'].map(step => <li key={step} className="rounded-lg bg-white/5 px-3 py-2 text-white/75">{step}</li>)}
        </ol>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium"><Pencil className="size-4 text-[#ffffff]" />{maskUrl ? '检查并修补蒙版' : '第一步：圈选对象'}</div>
            <p className="mt-1 truncate text-xs text-white/45">{fileName} · {message}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {testImages.length > 0 && <>
              <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={() => stepTestImage(-1)}>上一张</Button>
              <select value={testImageIndex} onChange={(event) => void chooseTestImage(Number(event.target.value))} className="h-10 max-w-56 rounded-lg border border-white/10 bg-[#181818] px-3 text-sm text-white outline-none focus:border-[#ededed]/60" aria-label="选择测试图片">
                {testImages.map((item, index) => <option key={item.name} value={index}>{item.name}</option>)}
              </select>
              <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={() => stepTestImage(1)}>下一张</Button>
            </>}
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={clearSelection} disabled={!lasso.length && !maskUrl}><RotateCcw /> 重新圈选</Button>
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={() => fileInputRef.current?.click()}><ImagePlus /> 换图片</Button>
            <input ref={fileInputRef} className="hidden" type="file" accept="image/png,image/jpeg,image/webp,.jpg,.jpeg" onChange={(event) => event.target.files?.[0] && chooseImage(event.target.files[0])} />
          </div>
        </div>

        <div className="grid items-start gap-4 lg:grid-cols-[220px_minmax(0,1fr)_280px]">
        <div className="relative flex min-h-[55vh] items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_50%_30%,#222222_0%,#101010_62%)] shadow-2xl lg:col-start-2 lg:row-start-1">
          <div className="relative max-h-[calc(100vh-13.5rem)] max-w-full select-none">
            {imageUrl && <img ref={imageRef} src={imageUrl} alt="待圈选图片" onLoad={onImageLoad} className="block max-h-[calc(100vh-13.5rem)] max-w-full object-contain" draggable={false} />}
            <canvas ref={canvasRef} aria-label="用画笔圈选对象" className="absolute inset-0 h-full w-full cursor-crosshair touch-none" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={finishLasso} onPointerCancel={finishLasso} />
          </div>
          {!lasso.length && !loading && <div className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-black/70 px-4 py-2 text-center text-sm text-white/75 shadow-xl backdrop-blur-md">按住鼠标或手指，沿对象外围画一圈</div>}
          {maskUrl && !loading && <div className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-[#ededed]/25 bg-black/75 px-4 py-2 text-center text-sm text-white/80 shadow-xl backdrop-blur-md">浅色蒙版必须完整覆盖对象；漏选用“添加”，误选用“擦除”</div>}
          {loading && <div className="absolute inset-0 grid place-items-center bg-black/25 backdrop-blur-[1px]"><div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#191919]/95 px-4 py-2 text-sm shadow-xl"><Spinner className="text-[#ededed]" /> 正在分割圈内对象</div></div>}
        </div>

        <aside className="flex flex-col gap-5 rounded-2xl border border-white/10 bg-white/[0.035] p-4 lg:col-start-1 lg:row-start-1">
          <h2 className="text-sm font-semibold">圈选与蒙版</h2>
          <p className="text-xs text-white/45">每次只圈一个主体 · 松手自动闭合 · 自动去除不相连的碎片、填孔和平滑</p>
          <Button variant="outline" onClick={clearSelection} disabled={loading || erasing}><Pencil />重新圈选</Button>
          {maskUrl && <div className="flex flex-col gap-2">
            <Button variant={editMode === 'add' ? 'default' : 'outline'} className={editMode === 'add' ? 'bg-[#ededed] text-[#151515] hover:bg-[#ffffff]' : 'border-white/10 bg-white/5 text-white hover:bg-white/10'} onClick={() => setEditMode('add')}><Paintbrush /> 添加蒙版</Button>
            <Button variant={editMode === 'erase' ? 'default' : 'outline'} className={editMode === 'erase' ? 'bg-[#dddddd] text-[#151515] hover:bg-[#ffffff]' : 'border-white/10 bg-white/5 text-white hover:bg-white/10'} onClick={() => setEditMode('erase')}><Eraser /> 擦除蒙版</Button>
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={undoMaskEdit} disabled={!canUndoMask}><Undo2 /> 撤销</Button>
            <Button variant="outline" className="border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={useLassoAsMask}>套索直接作为蒙版</Button>
          </div>}
          <div className="min-w-[180px] flex-1 sm:max-w-[290px]"><div className="mb-2 flex justify-between text-xs text-white/55"><span>蒙版透明度</span><span>{opacity}%</span></div><Slider value={[opacity]} min={15} max={80} step={1} onValueChange={(value) => setOpacity(sliderValue(value))} /></div>
          {!maskUrl && <div className="min-w-[180px] flex-1 sm:max-w-[290px]"><div className="mb-2 flex justify-between text-xs text-white/55"><span>分割搜索边距</span><span>{searchMargin}px</span></div><Slider value={[searchMargin]} min={0} max={192} step={8} onValueChange={(value) => setSearchMargin(sliderValue(value))} disabled={loading} /></div>}
          {maskUrl && <div><div className="mb-2 flex justify-between text-xs text-white/55"><span>修补画笔大小</span><span>{brushSize}px</span></div><Slider value={[brushSize]} min={8} max={160} step={4} onValueChange={(value) => setBrushSize(sliderValue(value))} /></div>}
          <p className="text-xs text-white/40">黄色：圈选 · 绿色：蒙版</p>
          {maskUrl && <div className="space-y-4 border-t border-white/10 pt-4">
            {([{label: '向内收缩', value: maskInward, change: setMaskInward}, {label: '向外扩展', value: maskOutward, change: setMaskOutward}, {label: '边缘平滑', value: maskSmooth, change: setMaskSmooth}]).map(item => <div key={item.label}><div className="mb-2 flex justify-between text-xs text-white/60"><span>{item.label}</span><span>{item.value}px</span></div><Slider value={[item.value]} min={0} max={16} step={1} onValueChange={v => item.change(sliderValue(v))} disabled={loading || erasing} /></div>)}
            <Button variant="outline" className="w-full" onClick={() => void cleanCurrentMask()} disabled={loading || erasing}>填孔并优化边缘</Button>
            <p className="text-xs text-white/40">按原图像素调整；每次点击应用一次。填孔会覆盖手臂与身体之间等封闭空隙。</p>
          </div>}
          <Button variant="outline" disabled={!maskUrl || loading || erasing || savingMask} onClick={() => void saveTestMask()}>{savingMask ? <Spinner /> : <Download className="size-4" />}{savingMask ? '保存中…' : '保存测试蒙版'}</Button>
          {maskSaveMessage && <p role="status" className="break-all text-xs text-white/60">{maskSaveMessage}</p>}
        </aside>

        <section className="grid gap-4 rounded-2xl border border-[#ededed]/20 bg-[#181818] p-4 lg:col-start-3 lg:row-start-1">
          <div>
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Eraser className="size-4 text-[#ededed]" />第二步：内容消除</div>
            <p className="mb-3 text-xs text-white/55">确认绿色区域完整覆盖目标；阴影需要消除时请一并涂入。仅重建无人背景，但模型仍可能生成新人物，请检查结果。</p>
            <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} className="min-h-64 border-white/10 bg-black/20 text-white" aria-label="消除提示词" />
          </div>
          <div className="flex flex-col justify-between gap-3">
            <div><div className="mb-2 flex justify-between text-xs text-white/55"><span>消除范围扩展</span><span>{maskExpand}px</span></div><Slider value={[maskExpand]} min={0} max={32} step={1} onValueChange={(value) => setMaskExpand(sliderValue(value))} disabled={erasing} /></div>
            <div className="space-y-3">
              <div>
                <label className="mb-1.5 flex items-center gap-2 text-sm text-white/70" htmlFor="content-backend"><Eraser className="size-4" />内容引擎</label>
                <select id="content-backend" disabled={erasing} value={contentBackend} onChange={(event) => setContentBackend(event.target.value as 'opencv' | 'comfyui-flux-fill')} className="h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 text-sm text-white outline-none focus:border-[#ededed]/60">
                  <option value="opencv" className="bg-[#181818]">OpenCV 重建</option>
                  <option value="comfyui-flux-fill" className="bg-[#181818]">FLUX.1 Fill-dev · Q4_K_S + 消除 LoRA v2</option>
                </select>
              </div>
              <label className="block text-xs text-white/65">SDR 生成步数
                <select value={sdrSteps} disabled={erasing || contentBackend === 'opencv'} onChange={e => setSdrSteps(Number(e.target.value))} className="mt-2 h-10 w-full rounded-lg bg-[#262626] px-3">
                  <option value={20}>20 步 · 当前基准</option><option value={12}>12 步 · 加速试验，可能影响质量</option>
                </select>
              </label>
              <label className="block text-xs text-white/65">Gainmap 方案
                <select value={gainmapMode} disabled={erasing} onChange={e => setGainmapMode(e.target.value)} className="mt-2 h-10 w-full rounded-lg bg-[#262626] px-3">
                  <option value="generative">生成式 · FLUX Fill + 消除 LoRA · 8 步</option>
                  <option value="smooth-interpolation">快速平滑插值 · 无额外模型调用</option>
                </select>
              </label>
              <p className="text-xs text-white/45">生成式为实验方案：最长边 576px、单通道、无 ControlNet / RGB 引导、无额外外扩或高斯模糊；需额外一次远程生成，不保证更快。普通 SDR 图片跳过此步骤。</p>
              {eraseResult?.timings_seconds && <p className="text-xs text-white/60">实际耗时：准备 {eraseResult.timings_seconds.preparation.toFixed(1)}s · SDR {eraseResult.timings_seconds.sdr.toFixed(1)}s · Gainmap / 合成 {eraseResult.timings_seconds.gainmap_and_encode.toFixed(1)}s</p>}
            </div>
            <div><Button className="h-11 w-full bg-[#ededed] text-[#151515] hover:bg-[#ffffff]" disabled={erasing || loading || !maskUrl} onClick={() => void runErase()}>{erasing ? <Spinner /> : <Sparkles />} {erasing ? '正在处理…' : '开始消除'}</Button>{erasing && <div className="mt-3 space-y-2"><div className="flex justify-between gap-2 text-xs text-white/75"><span>{eraseStage}</span><span>{eraseProgress}%</span></div><div role="progressbar" aria-label="预计消除进度" aria-valuenow={eraseProgress} aria-valuemin={0} aria-valuemax={100} className="h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-[#ededed] transition-all duration-300" style={{ width: `${eraseProgress}%` }} /></div><p className="text-xs text-white/40">阶段与百分比为估算 · {gainmapMode === 'generative' ? '生成式 HDR 通常约 2 分钟；普通 SDR 会跳过 Gainmap' : '耗时取决于选区与远程状态'} · 已用 {eraseElapsed} 秒</p></div>}{eraseMessage && <p className={`mt-2 text-xs ${eraseMessage.includes('完成') ? 'text-[#eeeeee]' : 'text-white/55'}`}>{eraseMessage}</p>}</div>
          </div>
        </section>
        </div>

        <section className="rounded-2xl border border-white/10 bg-[#181818] p-4">
          <h2 className="mb-2 font-semibold">测试样张</h2>
          <p className="mb-4 text-xs text-white/45">将图片放入 seg_ui/test_images 文件夹，刷新页面即可加入图集。</p>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {testImages.map((item, index) => <button key={item.name} disabled={erasing || loading} onClick={() => void chooseTestImage(index)} aria-pressed={testSourceName === item.name} className={`w-40 shrink-0 overflow-hidden rounded-xl border text-left disabled:opacity-40 ${testSourceName === item.name ? 'border-[#ededed] bg-[#ededed]/10' : 'border-white/10 bg-black/20'}`}>
              <img src={`${API_URL}${item.url}`} alt={item.name} loading="lazy" className="h-28 w-full object-cover" />
              <span className="block truncate px-3 py-2 text-xs">{item.name}</span>
            </button>)}
          </div>
        </section>

        <section className="space-y-4 rounded-2xl border border-white/10 bg-[#181818] p-4">
          <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">作品集</h2><Button onClick={() => void saveWork()} disabled={!currentWork || savingWork || erasing || loading}>{savingWork ? '保存中…' : '保存当前作品'}</Button></div>
          <p className="text-xs text-white/50">消除后自动替换编辑底图 · 保存完整 UHDR / SDR 到本地 seg_ui/portfolio · 点击作品可继续编辑</p>
          {workMessage && <p className="break-all text-xs text-white/70">{workMessage}</p>}
          <div className="flex gap-3 overflow-x-auto">{portfolio.map(work => <div key={work.id} className="w-40 shrink-0 rounded-lg border border-white/15 p-2"><button disabled={erasing || loading} onClick={() => void openWork(work)} className="w-full text-left"><img src={`${API_URL}${work.preview_url}`} alt={work.name} className="h-32 w-full rounded object-cover" /><p className="mt-2 text-xs">{work.name}</p><p className="text-xs text-white/50">{work.kind}</p></button><a href={`${API_URL}${work.url}`} download className="mt-2 block text-xs underline">下载完整文件</a></div>)}</div>
          {!portfolio.length && <p className="text-xs text-white/40">还没有保存的作品</p>}
        </section>

        <section className="space-y-4 rounded-2xl border border-white/10 bg-[#181818] p-4">
          <h2 className="font-semibold">消除前后对比</h2>
          {(['sdr', 'gainmap', 'hdr'] as const).map((kind) => {
            const pair = eraseResult?.comparisons?.[kind];
            return <div key={kind}>
              <h3 className="mb-2 text-sm font-medium">{kind === 'gainmap' ? 'Gainmap' : kind.toUpperCase()}</h3>
              <p className="mb-2 text-xs text-white/45">{kind === 'gainmap' ? '重建使用的单通道增益场 · 前后统一对数灰度，不是 JPEG 内嵌 gainmap 的直接导出' : kind === 'hdr' ? '原始 Ultra HDR 文件对比 · 实际高亮显示取决于浏览器与屏幕的 HDR 支持' : '相同尺寸的 SDR 图像'}</p>
              <div className="grid gap-3 sm:grid-cols-2">{['消除前', '消除后'].map((label, index) => <figure key={label} className="overflow-hidden rounded-xl border border-white/10 bg-black/25">
                <figcaption className="border-b border-white/10 px-3 py-2 text-xs text-white/60">{label}</figcaption>
                {pair ? <a href={`${API_URL}${pair[index]}`} target="_blank" rel="noreferrer"><img src={`${API_URL}${pair[index]}`} alt={`${kind} ${label}`} loading="lazy" className="max-h-[560px] w-full object-contain" /></a> : <div className="grid h-36 place-items-center text-sm text-white/35">{eraseResult ? '此图片无 HDR / gainmap 数据' : '完成一次消除后显示'}</div>}
              </figure>)}</div>
            </div>;
          })}
        </section>

        {eraseResult && <div className="flex items-center justify-between rounded-xl border border-white/10 p-4"><span className="text-sm text-white/60">消除完成 · {(eraseResult.elapsed_ms / 1000).toFixed(1)} 秒</span><a className="rounded-lg bg-[#ededed] px-4 py-2 text-sm text-[#151515]" href={`${API_URL}${eraseResult.download_url}`} download>下载 {eraseResult.kind}</a></div>}
      </section>
    </main>
  );
}
