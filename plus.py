"""
AR 手势阅读助手 - 开源模型版 v3.6.1

目标：
1. 保留原来的 Flask + MediaPipe + SSE + 前端 AR 交互结构。
2. 将图像理解从智谱 GLM-4V 替换为本地开源 Qwen2.5-VL。
3. 将 3D 生成从 TokenHub 替换为本地开源 TripoSR CLI。
4. 保留本地 3D 模型缓存，重复物体直接加载缓存。

推荐先单独跑通：
    Qwen2.5-VL：本地图像理解
    TripoSR：python run.py input.png --output-dir output --model-save-format glb

运行前建议配置环境变量：
PowerShell 示例：
    $env:HAND_MODEL_PATH="D:\\tool3\\py1\\pythonProject6\\hand_landmarker.task"
    $env:LOCAL_VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
    $env:TRIPOSR_DIR="D:\\tool3\\TripoSR"
    $env:TRIPOSR_PYTHON="D:\\tool3\\TripoSR\\venv\\Scripts\\python.exe"
    $env:LOCAL_3D_DEVICE="cuda:0"
    python ar_gesture_reader_open_source_v3_6_1_fixed.py

如果机器显存不够，可先设置：
    $env:LOCAL_VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
    $env:LOCAL_3D_DEVICE="cpu"

注意：
- Qwen2.5-VL 和 TripoSR 第一次运行会下载或加载模型，耗时较长。
- TripoSR 生成质量依赖 ROI 清晰度，最好框选单个物体，背景尽量简单。
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import edge_tts
import mediapipe as mp
import numpy as np
import requests
from flask import Flask, Response, send_from_directory
from mediapipe import tasks
from mediapipe.tasks.python import vision


# ==================== 日志 ====================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("ar-gesture-reader-open-source")

try:
    from cache_policy_loader import decide_cache_policy, load_cache_policy
    _POLICY_LOADER_AVAILABLE = True
    _POLICY_LOADER_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    decide_cache_policy = None  # type: ignore[assignment]
    load_cache_policy = None  # type: ignore[assignment]
    _POLICY_LOADER_AVAILABLE = False
    _POLICY_LOADER_IMPORT_ERROR = exc


# ==================== 基础配置 ====================

@dataclass(frozen=True)
class Config:
    model_path: str = os.getenv("HAND_MODEL_PATH", r"D:\tool3\py1\pythonProject6\hand_landmarker.task")

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "640"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "480"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "85"))

    trigger_frames: int = int(os.getenv("TRIGGER_FRAMES", "15"))
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "5"))
    pinch_threshold_norm: float = float(os.getenv("PINCH_THRESHOLD_NORM", "0.07"))
    min_roi_area_ratio: float = float(os.getenv("MIN_ROI_AREA_RATIO", "0.02"))
    gesture_event_interval: float = float(os.getenv("GESTURE_EVENT_INTERVAL", "0.35"))
    roi_multiframe_count: int = int(os.getenv("ROI_MULTIFRAME_COUNT", "12"))
    enable_roi_keyframe_preprocess: bool = os.getenv("ENABLE_ROI_KEYFRAME_PREPROCESS", "1") == "1"

    static_model_dir: Path = Path(os.getenv("STATIC_MODEL_DIR", "runtime_assets"))

    # 本地图像理解模型：Qwen2.5-VL
    local_vlm_model: str = os.getenv(
        "LOCAL_VLM_MODEL",
        r"D:\tool3\models\Qwen2.5-VL-3B-Instruct",
    )
    local_vlm_max_new_tokens: int = int(os.getenv("LOCAL_VLM_MAX_NEW_TOKENS", "160"))
    local_vlm_4bit: bool = os.getenv("LOCAL_VLM_4BIT", "0") == "1"

    # 本地 3D 生成：TripoSR
    triposr_dir: str = os.getenv("TRIPOSR_DIR", r"D:\tool3\TripoSR")
    triposr_python: str = os.getenv("TRIPOSR_PYTHON", r"D:\tool3\TripoSR\venv\Scripts\python.exe")
    local_3d_device: str = os.getenv(
        "LOCAL_3D_DEVICE",
        "cpu",
    )
    local_3d_timeout: int = int(os.getenv("LOCAL_3D_TIMEOUT", "600"))
    triposr_bake_texture: bool = os.getenv("TRIPOSR_BAKE_TEXTURE", "0") == "1"
    triposr_texture_resolution: int = int(os.getenv("TRIPOSR_TEXTURE_RESOLUTION", "1024"))
    enable_geometry_texture_split: bool = os.getenv("ENABLE_GEOMETRY_TEXTURE_SPLIT", "1") == "1"
    enable_progressive_model_loading: bool = os.getenv("ENABLE_PROGRESSIVE_MODEL_LOADING", "1") == "1"
    enable_policy_cache: bool = os.getenv("ENABLE_POLICY_CACHE", "0") == "1"
    cache_policy_path: str = os.getenv("CACHE_POLICY_PATH", "runtime_assets/cache_policy.json")


CONFIG = Config()
CONFIG.static_model_dir.mkdir(parents=True, exist_ok=True)
logger.info(
    "policy_cache_enabled=%s, cache_policy_path=%s",
    CONFIG.enable_policy_cache,
    CONFIG.cache_policy_path,
)

MODEL_CACHE_DIR = CONFIG.static_model_dir / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INDEX_PATH = MODEL_CACHE_DIR / "cache_index.json"
MODEL_SPLIT_DIR = MODEL_CACHE_DIR / "split_artifacts"
MODEL_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_3D_WORK_DIR = CONFIG.static_model_dir / "local_3d_work"
LOCAL_3D_WORK_DIR.mkdir(parents=True, exist_ok=True)

ROI_SEQUENCE_DIR = CONFIG.static_model_dir / "roi_sequences"
ROI_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

if CONFIG.enable_policy_cache and not _POLICY_LOADER_AVAILABLE:
    logger.warning("策略缓存已请求启用，但 cache_policy_loader 导入失败，已自动禁用：%s", _POLICY_LOADER_IMPORT_ERROR)


# ==================== Flask 和全局状态 ====================

app = Flask(__name__)
sse_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=200)
processing_lock = threading.Lock()

# Qwen 模型懒加载缓存
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_LOCK = threading.Lock()


# ==================== 前端页面 ====================

HTML_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>AR 手势阅读助手 - 开源模型版</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:#000;font-family:Arial,"Microsoft YaHei",sans-serif}#container{position:relative;width:100vw;height:100vh;overflow:hidden;background:#000}#video-feed{width:100%;height:100%;object-fit:cover}@media(min-width:768px){#video-feed{width:auto;height:auto;max-width:min(100%,900px);max-height:min(100%,680px);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);object-fit:contain;background:#000;border-radius:12px}}#ar-model-layer{position:absolute;left:50%;top:50%;width:260px;height:260px;transform:translate(-50%,-50%) scale(1) rotateY(0deg);z-index:18;display:none;pointer-events:auto;touch-action:none}#ar-model-layer.show{display:block}#ar-model-container,#ar-model-container model-viewer{width:100%;height:100%;background:transparent}#status-bar{position:absolute;top:10px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.64);color:#fff;padding:7px 18px;border-radius:20px;font-size:14px;pointer-events:none;z-index:30;backdrop-filter:blur(4px);max-width:92vw;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}#gesture-popup{position:absolute;bottom:84px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.78);color:#fff;padding:12px 24px;border-radius:30px;font-size:16px;font-weight:bold;pointer-events:none;z-index:30;opacity:0;transition:opacity .25s;backdrop-filter:blur(4px)}#gesture-popup.show{opacity:1}#btn-group{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);display:flex;gap:10px;z-index:40;flex-wrap:wrap;justify-content:center}#btn-group button{padding:10px 18px;border:none;border-radius:25px;background:rgba(255,255,255,.2);color:#fff;font-size:14px;cursor:pointer;backdrop-filter:blur(4px);transition:background .2s}#btn-group button:active,#btn-group button.touch-active{background:rgba(255,255,255,.45)}#info-card{position:absolute;right:14px;top:58px;width:min(350px,calc(100vw - 28px));max-height:calc(100vh - 150px);overflow:auto;background:rgba(0,0,0,.62);color:#fff;border:1px solid rgba(255,255,255,.16);border-radius:16px;padding:12px;z-index:25;backdrop-filter:blur(6px);display:none}#info-card.show{display:block}#knowledge{line-height:1.55;font-size:14px;margin-bottom:10px;white-space:pre-wrap}#model-link{display:none;color:#9ad1ff;font-size:13px;margin-top:8px;word-break:break-all}#model-link.show{display:block}#hint{margin-top:8px;font-size:12px;line-height:1.45;color:rgba(255,255,255,.72)}#badge{display:inline-block;margin-top:8px;padding:3px 8px;border-radius:999px;background:rgba(80,180,255,.18);border:1px solid rgba(120,210,255,.25);font-size:12px;color:#c9efff}#review-card{position:absolute;left:14px;top:58px;width:min(340px,calc(100vw - 28px));background:rgba(8,14,18,.78);color:#fff;border:1px solid rgba(120,210,255,.26);border-radius:12px;padding:12px;z-index:28;backdrop-filter:blur(6px);display:none;box-shadow:0 8px 24px rgba(0,0,0,.22)}#review-card.show{display:block}.review-title{font-size:15px;font-weight:bold;margin-bottom:6px}.review-body{font-size:13px;line-height:1.5;color:rgba(255,255,255,.86)}.review-meta{margin-top:8px;font-size:12px;color:#c9efff}.review-actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.review-actions button{border:1px solid rgba(255,255,255,.18);border-radius:8px;padding:7px 10px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.62)}
</style>
</head>
<body>
<div id="container">
    <img id="video-feed" src="/video_feed" alt="摄像头画面">
    <div id="ar-model-layer"><div id="ar-model-container"></div></div>
    <div id="status-bar">🖐️ 等待手势...</div>
    <div id="gesture-popup"></div>
    <div id="info-card">
        <div id="knowledge">等待 AI 识别结果...</div>
        <a id="model-link" target="_blank" rel="noopener">打开 3D 模型文件</a>
        <div id="badge">开源模型版：Qwen2.5-VL + TripoSR</div>
        <div id="hint">交互提示：一只手捏合拖动模型；两只手捏合缩放模型；左右滑动旋转模型。</div>
    </div>
    <div id="review-card" aria-live="polite">
        <div class="review-title">发现相似缓存模型</div>
        <div id="review-body" class="review-body">系统检测到一个相似缓存模型，后续可支持一键复用。当前实验阶段仍将继续原生成流程。</div>
        <div id="review-meta" class="review-meta"></div>
        <div class="review-actions"><button disabled>复用缓存模型（后续开放）</button><button disabled>继续生成</button></div>
    </div>
    <div id="btn-group">
        <button id="btn-scroll-up">⬆ 上滚</button><button id="btn-scroll-down">⬇ 下滚</button><button id="btn-zoom-in">🔍+</button><button id="btn-zoom-out">🔍-</button>
    </div>
</div>
<script>
const videoFeed=document.getElementById('video-feed');const gesturePopup=document.getElementById('gesture-popup');const statusBar=document.getElementById('status-bar');const infoCard=document.getElementById('info-card');const knowledge=document.getElementById('knowledge');const modelLink=document.getElementById('model-link');const arModelLayer=document.getElementById('ar-model-layer');const arModelContainer=document.getElementById('ar-model-container');const reviewCard=document.getElementById('review-card');const reviewBody=document.getElementById('review-body');const reviewMeta=document.getElementById('review-meta');let popupTimer=null,videoReloadTimer=null,videoLastLoad=Date.now(),modelScale=1,modelX=.5,modelY=.5,modelRotationY=0,modelLoaded=false,sseErrorTimer=null,sseWasDisconnected=false;function reloadVideoFeed(reason){if(!videoFeed)return;console.warn('?????:',reason);videoFeed.src='/video_feed?t='+Date.now()}if(videoFeed){videoFeed.onload=function(){videoLastLoad=Date.now();if(videoReloadTimer){clearTimeout(videoReloadTimer);videoReloadTimer=null}};videoFeed.onerror=function(){reloadVideoFeed('img_error')};setInterval(()=>{if(Date.now()-videoLastLoad>45000)reloadVideoFeed('watchdog_timeout')},30000)}const eventSource=new EventSource('/events');eventSource.onopen=function(){if(sseErrorTimer){clearTimeout(sseErrorTimer);sseErrorTimer=null}if(sseWasDisconnected){sseWasDisconnected=false;setStatus('??? ????...');console.log('SSE ???')}};eventSource.onmessage=function(e){try{handleServerEvent(JSON.parse(e.data))}catch(err){console.error('SSE JSON parse error:',err,e.data)}};eventSource.onerror=function(){console.warn('SSE ?????????????');if(sseErrorTimer)return;sseErrorTimer=setTimeout(()=>{sseWasDisconnected=true;statusBar.textContent='?? ??????????????...'},20000)};function handleServerEvent(data){switch(data.type){case'gesture':handleGesture(data);break;case'drag':handleDrag(data);break;case'scale':handleScale(data);break;case'status':setStatus(data.text||'处理中...');showPopup(data.text||'处理中...');break;case'knowledge':showKnowledge(data.text||'');break;case'audio':playAudio(data.url);break;case'model':showModel(data.url,data);break;case'cache_review':showCacheReview(data);break;case'error':setStatus('❌ '+(data.text||'发生错误'));showPopup(data.text||'发生错误');break;default:console.debug('未处理事件:',data)}}function showCacheReview(data){if(!reviewCard)return;const score=data.policy_score==null?'--':Number(data.policy_score).toFixed(2);const weak=data.weak_threshold==null?'--':data.weak_threshold;const strong=data.strong_threshold==null?'--':data.strong_threshold;reviewBody.textContent=data.text||'系统检测到一个相似缓存模型，后续可支持一键复用。当前实验阶段仍将继续原生成流程。';reviewMeta.textContent=`候选：${data.best_keyword||'未知'} ｜ score=${score} ｜ weak=${weak} ｜ strong=${strong}`;reviewCard.classList.add('show');setStatus('发现相似缓存模型，当前仍继续原生成流程');showPopup('发现相似缓存模型，后续可支持复用')}function setStatus(text){statusBar.textContent=text}function handleGesture(data){const gesture=data.gesture,detail=data.detail,confidence=data.confidence;const percent=confidence?` (${Math.round(confidence*100)}%)`:'';let text='';switch(gesture){case'open_palm':text='🖐️ 张开手掌 — 待命中';break;case'point_up':text='👆 食指上指 — 向上滚动';break;case'point_down':text='👇 食指下指 — 向下滚动';break;case'fist':text='✊ 握拳 — 确认/选中';break;case'two_fingers_up':text='✌️ 两指上指 — 放大';break;case'two_fingers_down':text='🤞 两指下指 — 缩小';break;case'swipe_left':text='👈 左滑 — 旋转模型';break;case'swipe_right':text='👉 右滑 — 旋转模型';break;default:text='🤔 未识别手势'}setStatus(text+percent);if(detail)showPopup(detail);executeAction(gesture)}function showPopup(msg){if(!msg)return;if(popupTimer)clearTimeout(popupTimer);gesturePopup.textContent=msg;gesturePopup.classList.add('show');popupTimer=setTimeout(()=>gesturePopup.classList.remove('show'),1100)}function executeAction(gesture){window.parent.postMessage({source:'ar-gesture-reader',gesture:gesture},'*');window.dispatchEvent(new CustomEvent('ar-gesture',{detail:{gesture:gesture}}));switch(gesture){case'point_up':window.scrollBy({top:-120,behavior:'smooth'});break;case'point_down':window.scrollBy({top:120,behavior:'smooth'});break;case'two_fingers_up':handleScale({factor:1.08});break;case'two_fingers_down':handleScale({factor:.92});break;case'swipe_left':rotateModel(-20);break;case'swipe_right':rotateModel(20);break}}function handleDrag(data){if(!data.active||!modelLoaded)return;modelX=Number(data.x||modelX);modelY=Number(data.y||modelY);updateArModelTransform();window.dispatchEvent(new CustomEvent('ar-drag',{detail:data}))}function handleScale(data){if(!modelLoaded)return;const factor=Number(data.factor||1);modelScale=Math.min(3.2,Math.max(.35,modelScale*factor));updateArModelTransform();const viewer=arModelContainer.querySelector('model-viewer');if(viewer)viewer.setAttribute('scale',`${modelScale} ${modelScale} ${modelScale}`);window.dispatchEvent(new CustomEvent('ar-scale',{detail:{factor:factor,modelScale:modelScale}}))}function rotateModel(delta){if(!modelLoaded)return;modelRotationY+=delta;updateArModelTransform()}function updateArModelTransform(){if(!arModelLayer)return;const x=modelX*window.innerWidth,y=modelY*window.innerHeight;arModelLayer.style.left=`${x}px`;arModelLayer.style.top=`${y}px`;arModelLayer.style.transform=`translate(-50%, -50%) scale(${modelScale}) rotateY(${modelRotationY}deg)`}function showKnowledge(text){infoCard.classList.add('show');knowledge.textContent=text||'没有识别到有效内容'}function playAudio(url){if(!url)return;const audio=new Audio(url+'?t='+Date.now());audio.play().catch(err=>console.warn('浏览器阻止自动播放，需要用户先点击页面:',err))}function showModel(url,meta={}){if(!url)return;infoCard.classList.add('show');modelLink.href=url;modelLink.textContent='打开 3D 模型文件';modelLink.classList.add('show');arModelLayer.classList.add('show');const started=performance.now();arModelContainer.innerHTML=`<model-viewer src="${url}?t=${Date.now()}" camera-controls auto-rotate ar exposure="1" shadow-intensity="0.7" interaction-prompt="none"></model-viewer>`;const viewer=arModelContainer.querySelector('model-viewer');if(viewer){viewer.addEventListener('load',()=>{const ms=Math.round(performance.now()-started);console.log('模型加载完成:',{load_ms:ms,cache_hit:meta.cache_hit||false,keyword:meta.keyword||'',url:url})},{once:true})}modelX=.5;modelY=.5;modelScale=1;modelRotationY=0;modelLoaded=true;updateArModelTransform();if(meta.cache_hit){setStatus('✅ 已从缓存加载 3D 模型');showPopup('缓存命中，模型已快速加载')}else{setStatus('✅ 3D 模型已出现在画面中');showPopup('3D 模型已出现，可用手势互动')}}function showModel(url,meta={}){if(!url)return;infoCard.classList.add('show');const geometryUrl=meta.geometry_url||'';const fullUrl=meta.full_url||url;const firstUrl=geometryUrl||fullUrl;modelLink.href=fullUrl;modelLink.textContent='打开 3D 模型文件';modelLink.classList.add('show');arModelLayer.classList.add('show');modelX=.5;modelY=.5;modelScale=1;modelRotationY=0;modelLoaded=true;const started=performance.now();let geometryVisibleMs=null;function mountModel(src,phase,onLoaded){arModelContainer.innerHTML=`<model-viewer src="${src}?t=${Date.now()}" camera-controls auto-rotate ar exposure="1" shadow-intensity="0.7" interaction-prompt="none"></model-viewer>`;const viewer=arModelContainer.querySelector('model-viewer');if(viewer){viewer.addEventListener('load',()=>{const ms=Math.round(performance.now()-started);console.log('模型加载阶段:',{phase:phase,load_ms:ms,geometry_first_visible_ms:geometryVisibleMs,cache_hit:meta.cache_hit||false,keyword:meta.keyword||'',url:src});if(onLoaded)onLoaded(ms)},{once:true})}updateArModelTransform()}if(geometryUrl&&geometryUrl!==fullUrl){mountModel(geometryUrl,'geometry',ms=>{geometryVisibleMs=ms;setStatus('✅ 几何模型已先显示，正在加载完整外观');setTimeout(()=>mountModel(fullUrl,'full',()=>{setStatus('✅ 完整 3D 模型已出现');showPopup('完整模型已替换')}),60)})}else{mountModel(firstUrl,'full',null)}updateArModelTransform();if(meta.cache_hit){setStatus(geometryUrl?'✅ 缓存命中，先显示几何模型':'✅ 已从缓存加载 3D 模型');showPopup('缓存命中，模型已快速加载')}else{setStatus(geometryUrl?'✅ 3D 几何模型先行显示':'✅ 3D 模型已出现在画面中');showPopup('3D 模型已出现，可用手势互动')}}function addButtonListeners(id,gestureName){const btn=document.getElementById(id);if(!btn)return;['mousedown','touchstart'].forEach(evt=>btn.addEventListener(evt,function(e){e.preventDefault();btn.classList.add('touch-active');executeAction(gestureName)}));['mouseup','mouseleave','touchend','touchcancel'].forEach(evt=>btn.addEventListener(evt,function(){btn.classList.remove('touch-active')}))}addButtonListeners('btn-scroll-up','point_up');addButtonListeners('btn-scroll-down','point_down');addButtonListeners('btn-zoom-in','two_fingers_up');addButtonListeners('btn-zoom-out','two_fingers_down');window.addEventListener('resize',updateArModelTransform);console.log('✅ AR 手势阅读助手开源模型版前端已就绪');
</script>
</body></html>
'''


# ==================== 通用工具函数 ====================

def push_event(event: Dict[str, Any]) -> None:
    try:
        sse_queue.put_nowait(event)
    except queue.Full:
        try:
            sse_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sse_queue.put_nowait(event)
        except queue.Full:
            logger.warning("SSE 队列已满，丢弃事件：%s", event.get("type"))


def sse_pack(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def make_message_jpeg(message: str, width: int = 640, height: int = 480) -> bytes:
    frame = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(frame, message[:42], (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if ok else b""


def dist_norm(a: Any, b: Any) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def dist_px(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


HAND_LINES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# ==================== 缓存工具 ====================

def load_cache_index() -> Dict[str, Dict[str, Any]]:
    if not CACHE_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache_index(index: Dict[str, Dict[str, Any]]) -> None:
    CACHE_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_keyword(keyword: str) -> str:
    keyword = (keyword or "").strip()
    for word in ["一个", "一只", "一台", "可爱的", "有趣的", "卡通", "模型", "玩具", "小朋友", "儿童", "物体"]:
        keyword = keyword.replace(word, "")
    keyword = keyword.strip()
    aliases = {
        "小恐龙": "恐龙", "恐龙玩偶": "恐龙", "恐龙玩具": "恐龙",
        "地球仪模型": "地球仪", "玩具熊": "熊", "泰迪熊": "熊",
        "小汽车": "汽车", "汽车玩具": "汽车", "火箭模型": "火箭", "飞机模型": "飞机",
    }
    return aliases.get(keyword, keyword or "未知物体")


def safe_filename(name: str) -> str:
    name = normalize_keyword(name)
    visible = "".join(c for c in name if c.isalnum() or c in "_-")[:24] or "object"
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:10]
    return f"{visible}_{digest}"


def cached_model_path(keyword: str) -> Path:
    return MODEL_CACHE_DIR / f"{safe_filename(keyword)}.glb"


def find_cached_model(keyword: str) -> Optional[Path]:
    path = cached_model_path(keyword)
    return path if path.exists() and path.stat().st_size > 0 else None


def record_cache(keyword: str, model_path: Path, source: str) -> None:
    keyword_norm = normalize_keyword(keyword)
    index = load_cache_index()
    index[keyword_norm] = {
        "keyword": keyword_norm,
        "filename": model_path.name,
        "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
    }
    save_cache_index(index)


def split_artifact_dir(keyword: str) -> Path:
    return MODEL_SPLIT_DIR / safe_filename(keyword)


def split_report_path(keyword: str) -> Path:
    return split_artifact_dir(keyword) / "split_report.json"


def model_url_for_static_path(path: Path) -> str:
    rel = Path(path).resolve().relative_to(CONFIG.static_model_dir.resolve())
    return "/models/" + rel.as_posix()


def progressive_model_meta(keyword: str, model_path: Path) -> Dict[str, str]:
    if not CONFIG.enable_progressive_model_loading:
        return {}

    artifact_dir = split_artifact_dir(keyword)
    geometry_path = artifact_dir / "model_geometry_only.glb"
    full_path = artifact_dir / "model_full_textured.glb"

    if not geometry_path.exists():
        return {}

    return {
        "geometry_url": model_url_for_static_path(geometry_path),
        "full_url": model_url_for_static_path(full_path if full_path.exists() else model_path),
        "progressive": True,
    }


def update_cache_split_metadata(keyword: str, split_report_path: Path, split_success: bool) -> None:
    keyword_norm = normalize_keyword(keyword)
    index = load_cache_index()
    item = index.get(keyword_norm, {"keyword": keyword_norm})
    item["split_enabled"] = CONFIG.enable_geometry_texture_split
    item["split_success"] = split_success
    item["split_report"] = str(split_report_path)
    item["split_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    index[keyword_norm] = item
    save_cache_index(index)


def split_model_for_cache(keyword: str, model_path: Path) -> None:
    if not CONFIG.enable_geometry_texture_split:
        return

    output_dir = split_artifact_dir(keyword)
    report_path = split_report_path(keyword)

    try:
        from paper_geometry_texture_split import split_geometry_texture

        report = split_geometry_texture(model_path, output_dir)
        update_cache_split_metadata(keyword, report_path, bool(report.success))
        if report.success:
            logger.info("Geometry/texture split completed: %s", report_path)
        else:
            logger.warning("Geometry/texture split failed: %s", report.error)
    except Exception as exc:
        logger.exception("Geometry/texture split post-process failed: %s", exc)
        update_cache_split_metadata(keyword, report_path, False)


def start_split_model_for_cache(keyword: str, model_path: Path) -> None:
    if not CONFIG.enable_geometry_texture_split:
        return

    threading.Thread(
        target=split_model_for_cache,
        args=(keyword, model_path),
        daemon=True,
    ).start()


def find_fused_cached_model(
    keyword: str,
    explanation: str,
    query_image: Path,
    threshold: float = 0.82,
) -> Optional[Tuple[Path, float, str]]:
    try:
        from cache_similarity import build_similarity_index, score_cache_entries

        index_path = MODEL_CACHE_DIR / "cache_similarity_index.json"
        entries = build_similarity_index(
            cache_dir=MODEL_CACHE_DIR,
            reference_dir=MODEL_CACHE_DIR / "reference_images",
            output_path=index_path,
        )
        query_text = f"{keyword} {explanation}".strip()
        results = score_cache_entries(
            entries,
            query_text=query_text,
            query_image=Path(query_image),
            threshold=threshold,
        )
        if not results:
            return None

        best = results[0]
        model_path = Path(best.model_path)
        if best.fused_score >= threshold and model_path.exists():
            return model_path, float(best.fused_score), best.keyword

        return None

    except Exception as exc:
        logger.warning("融合缓存判断失败，回退到 TripoSR：%s", exc)
        return None


def find_fused_cached_model_with_policy_meta(
    keyword: str,
    explanation: str,
    query_image: Path,
) -> Optional[Dict[str, Any]]:
    try:
        from cache_similarity import build_similarity_index, score_cache_entries

        index_path = MODEL_CACHE_DIR / "cache_similarity_index.json"
        entries = build_similarity_index(
            cache_dir=MODEL_CACHE_DIR,
            reference_dir=MODEL_CACHE_DIR / "reference_images",
            output_path=index_path,
        )
        query_text = f"{keyword} {explanation}".strip()
        results = score_cache_entries(
            entries,
            query_text=query_text,
            query_image=Path(query_image),
            text_weight=0.5,
            image_weight=0.5,
            threshold=0.7,
        )
        if not results:
            return None

        best = results[0]
        model_path = Path(best.model_path)
        return {
            "best_model_path": model_path,
            "best_keyword": best.keyword,
            "best_filename": best.filename,
            "text_score": best.text_score,
            "image_score": best.image_score,
            "fused_score": best.fused_score,
        }
    except Exception as exc:
        logger.warning("融合缓存 policy metadata 获取失败，回退原流程：%s", exc)
        return None


def decide_policy_cache_from_meta(meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "policy_cache_enabled": CONFIG.enable_policy_cache,
        "fallback_reason": "",
        "best_model_path": str(meta.get("best_model_path") or ""),
        "text_score": meta.get("text_score"),
        "image_score": meta.get("image_score"),
    }

    if not CONFIG.enable_policy_cache:
        info["fallback_reason"] = "policy_cache_disabled"
        return "miss", info
    if not _POLICY_LOADER_AVAILABLE or load_cache_policy is None or decide_cache_policy is None:
        info["fallback_reason"] = "policy_loader_unavailable"
        return "miss", info

    try:
        policy = load_cache_policy(CONFIG.cache_policy_path)
        decision = decide_cache_policy(meta.get("text_score"), meta.get("image_score"), policy)
        info.update(decision)
        return str(decision.get("policy_decision") or "miss"), info
    except Exception as exc:
        info["fallback_reason"] = f"policy_error: {exc}"
        logger.warning("策略缓存判断失败，回退原流程：%s", exc)
        return "miss", info


# ==================== 开源图像理解：Qwen2.5-VL ====================

def _load_qwen_vl():
    """懒加载 Qwen2.5-VL。首次调用会比较慢。"""
    global _QWEN_MODEL, _QWEN_PROCESSOR

    with _QWEN_LOCK:
        if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
            return _QWEN_MODEL, _QWEN_PROCESSOR

        push_event({"type": "status", "text": "正在加载本地 Qwen2.5-VL 模型..."})
        logger.info("加载本地 VLM：%s", CONFIG.local_vlm_model)

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            kwargs: Dict[str, Any] = {"device_map": "auto"}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16
                kwargs["attn_implementation"] = "sdpa"
            else:
                kwargs["torch_dtype"] = torch.float32

            if CONFIG.local_vlm_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                except Exception as exc:
                    logger.warning("4bit 量化不可用，改用普通加载：%s", exc)

            _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(CONFIG.local_vlm_model, **kwargs)
            _QWEN_PROCESSOR = AutoProcessor.from_pretrained(CONFIG.local_vlm_model)
            logger.info("Qwen2.5-VL 加载完成")
            return _QWEN_MODEL, _QWEN_PROCESSOR

        except Exception as exc:
            logger.exception("Qwen2.5-VL 加载失败：%s", exc)
            raise


def _parse_vlm_output(text: str) -> Tuple[str, str]:
    text = (text or "").strip()

    # 优先解析 JSON
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            explanation = str(obj.get("explanation") or obj.get("讲解") or "").strip()
            keyword = str(obj.get("keyword") or obj.get("关键词") or obj.get("object") or "").strip()
            if explanation and keyword:
                return explanation, normalize_keyword(keyword)
    except Exception:
        pass

    # 兼容 “讲解|关键词” 格式
    if "|" in text:
        explanation, keyword = text.rsplit("|", 1)
        return explanation.strip(), normalize_keyword(keyword.strip())

    # 兜底：取前一句作为说明，关键词使用通用值
    return text[:120] or "这是一个有趣的物体。", "可爱的卡通角色"


def analyze_image_local_qwen(image_path: Path) -> str:
    """
    本地开源图像理解：ROI 图片 -> 儿童化讲解文本 + 关键词。
    返回仍保持 explanation|keyword，方便复用原流程。
    """
    try:
        import torch
        from PIL import Image

        model, processor = _load_qwen_vl()
        image = Image.open(image_path).convert("RGB")

        prompt = (
            "请观察图片中的主要物体。你是一位幼儿科普老师，请输出严格 JSON，不要添加额外文字。"
            "JSON 格式为：{\"explanation\":\"60字以内的儿童化中文讲解\",\"keyword\":\"最适合3D建模的简短中文物体名\"}。"
            "如果看不清，请给出最可能的类别。"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=CONFIG.local_vlm_max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        explanation, keyword = _parse_vlm_output(output_text)
        logger.info("本地 VLM 输出：%s | %s", explanation, keyword)
        return f"{explanation}|{keyword}"

    except Exception as exc:
        logger.exception("本地 Qwen2.5-VL 图像理解失败：%s", exc)
        return "这是一个有趣的物体，我还没能准确识别它。|可爱的卡通角色"


# ==================== 语音合成 ====================

async def tts_to_file(text: str, filename: Path) -> Path:
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(str(filename))
    return filename


def speak_and_push(text: str) -> None:
    def run() -> None:
        filename = CONFIG.static_model_dir / f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp3"
        try:
            asyncio.run(tts_to_file(text, filename))
            push_event({"type": "audio", "url": f"/models/{filename.name}"})
        except Exception as exc:
            logger.exception("TTS 生成失败：%s", exc)
            push_event({"type": "error", "text": "语音生成失败"})

    threading.Thread(target=run, daemon=True).start()


def save_roi_sequence(frames: List[Any]) -> Path:
    sequence_dir = ROI_SEQUENCE_DIR / f"roi_seq_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    for idx, roi in enumerate(frames):
        frame_path = sequence_dir / f"roi_{idx:03d}.jpg"
        cv2.imwrite(str(frame_path), roi)

    return sequence_dir


def prepare_roi_input(roi_input: Path) -> Path:
    if not CONFIG.enable_roi_keyframe_preprocess or not Path(roi_input).is_dir():
        return roi_input

    run_id = f"roi_prepare_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_root = CONFIG.static_model_dir / "roi_preprocess_outputs" / run_id
    keyframe_dir = output_root / "keyframes"
    preprocess_dir = output_root / "foreground"

    try:
        from paper_keyframe_repro import run_keyframe_selection

        push_event({"type": "status", "text": "正在从多帧 ROI 中选择最佳帧..."})
        best_frame = run_keyframe_selection(
            input_dir=roi_input,
            output_dir=keyframe_dir,
            max_keyframes=min(5, max(1, CONFIG.roi_multiframe_count)),
        )
        if not best_frame:
            raise RuntimeError("keyframe selection returned no best frame")

        try:
            from paper_preprocess_repro import preprocess_foreground

            push_event({"type": "status", "text": "正在进行前景 mask 与 bbox 预处理..."})
            report = preprocess_foreground(
                input_path=best_frame,
                output_dir=preprocess_dir,
                white_background=True,
            )
            clean_path = Path(report.clean_image)
            if report.success and clean_path.exists():
                logger.info("ROI 多帧预处理完成：%s", clean_path)
                return clean_path
            logger.warning("ROI 前景预处理未成功，回退到 best_frame：%s", report.error)
            return best_frame
        except Exception as exc:
            logger.exception("ROI 前景预处理异常，回退到 best_frame：%s", exc)
            return best_frame

    except Exception as exc:
        logger.exception("ROI 多帧关键帧选择异常，回退到第一帧：%s", exc)
        candidates = sorted(Path(roi_input).glob("*.jpg"))
        return candidates[0] if candidates else roi_input


# ==================== 开源 3D 生成：TripoSR CLI ====================
def generate_3d_model_local_triposr(roi_path: Path, keyword: str) -> Tuple[Optional[Path], Optional[str]]:
    """
    使用本地 TripoSR 生成 GLB。
    修复 Windows 下 subprocess 日志 GBK 解码失败的问题。
    """
    if not CONFIG.triposr_dir:
        return None, "未配置 TRIPOSR_DIR，无法调用本地 TripoSR"

    triposr_dir = Path(CONFIG.triposr_dir)
    run_py = triposr_dir / "run.py"

    if not run_py.exists():
        return None, f"未找到 TripoSR run.py：{run_py}"

    out_dir = LOCAL_3D_WORK_DIR / f"{safe_filename(keyword)}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        CONFIG.triposr_python,
        str(run_py),
        str(roi_path.resolve()),
        "--output-dir",
        str(out_dir.resolve()),
        "--model-save-format",
        "glb",
        "--device",
        CONFIG.local_3d_device,
    ]

    if CONFIG.triposr_bake_texture:
        cmd += [
            "--bake-texture",
            "--texture-resolution",
            str(CONFIG.triposr_texture_resolution),
        ]

    logger.info("调用 TripoSR：%s", " ".join(cmd))
    push_event({
        "type": "status",
        "text": "本地 TripoSR 生成 3D 模型中..."
    })

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(triposr_dir.resolve()),
            capture_output=True,
            text=False,          # 关键：不要让 Windows 自动用 GBK 解码
            timeout=CONFIG.local_3d_timeout,
            check=False,
        )

        stdout_text = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
        stderr_text = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""

        if completed.returncode != 0:
            logger.error("TripoSR returncode: %s", completed.returncode)
            logger.error("TripoSR stdout:\n%s", stdout_text[-3000:])
            logger.error("TripoSR stderr:\n%s", stderr_text[-3000:])
            return None, "TripoSR 生成失败，请检查终端日志和依赖环境"

        glb_candidates = list(out_dir.rglob("*.glb"))

        if not glb_candidates:
            logger.error("TripoSR 未找到 GLB 输出")
            logger.error("TripoSR stdout:\n%s", stdout_text[-3000:])
            logger.error("TripoSR stderr:\n%s", stderr_text[-3000:])
            return None, "TripoSR 未输出 GLB 文件"

        generated_glb = glb_candidates[0]
        cache_path = cached_model_path(keyword)

        shutil.copy2(generated_glb, cache_path)
        record_cache(keyword, cache_path, source="TripoSR")
        split_model_for_cache(keyword, cache_path)

        logger.info("TripoSR 模型已缓存：%s", cache_path)

        return cache_path, None

    except subprocess.TimeoutExpired as exc:
        stdout_text = ""

        if exc.stdout:
            if isinstance(exc.stdout, bytes):
                stdout_text = exc.stdout.decode("utf-8", errors="replace")
            else:
                stdout_text = str(exc.stdout)

        stderr_text = ""

        if exc.stderr:
            if isinstance(exc.stderr, bytes):
                stderr_text = exc.stderr.decode("utf-8", errors="replace")
            else:
                stderr_text = str(exc.stderr)

        logger.error("TripoSR 生成超时")
        logger.error("TripoSR stdout:\n%s", stdout_text[-3000:])
        logger.error("TripoSR stderr:\n%s", stderr_text[-3000:])

        return None, "TripoSR 生成超时"

    except Exception as exc:
        logger.exception("TripoSR 调用异常：%s", exc)
        return None, "TripoSR 调用异常"

# ==================== 处理 ROI：开源 VLM + 本地 3D + 缓存 ====================

def process_roi_async(roi_path: Path) -> None:
    def run() -> None:
        try:
            push_event({"type": "status", "text": "本地开源模型识别中..."})
            roi_path_for_model = prepare_roi_input(roi_path)

            result = analyze_image_local_qwen(roi_path_for_model)
            if "|" in result:
                explanation, keyword = result.rsplit("|", 1)
            else:
                explanation = result
                keyword = "可爱的卡通角色"

            explanation = explanation.strip() or "这是一个有趣的物体！"
            keyword = normalize_keyword(keyword.strip() or "可爱的卡通角色")

            push_event({"type": "knowledge", "text": explanation})
            speak_and_push(explanation)

            # 缓存命中：直接加载
            cached_path = find_cached_model(keyword)
            if cached_path:
                if CONFIG.enable_geometry_texture_split and not split_report_path(keyword).exists():
                    split_model_for_cache(keyword, cached_path)
                logger.info("模型缓存命中：%s -> %s", keyword, cached_path)
                push_event({"type": "status", "text": f"缓存命中：{keyword}"})
                push_event({
                    "type": "model",
                    "url": f"/models/model_cache/{cached_path.name}",
                    "keyword": keyword,
                    "cache_hit": True,
                    **progressive_model_meta(keyword, cached_path),
                })
                return

            query_image_for_cache = roi_path_for_model if Path(roi_path_for_model).exists() else roi_path
            if CONFIG.enable_policy_cache:
                fused_meta = find_fused_cached_model_with_policy_meta(
                    keyword=keyword,
                    explanation=explanation,
                    query_image=query_image_for_cache,
                )
                if fused_meta:
                    policy_decision, policy_info = decide_policy_cache_from_meta(fused_meta)
                    policy_path = Path(fused_meta["best_model_path"])
                    fused_keyword = str(fused_meta.get("best_keyword") or keyword)
                    logger.info(
                        "策略缓存判断：enabled=%s, policy_score=%s, decision=%s, text_score=%s, "
                        "image_score=%s, weak=%s, strong=%s, best_model_path=%s, fallback_reason=%s",
                        policy_info.get("policy_cache_enabled"),
                        policy_info.get("policy_score"),
                        policy_decision,
                        policy_info.get("text_score"),
                        policy_info.get("image_score"),
                        policy_info.get("weak_threshold"),
                        policy_info.get("strong_threshold"),
                        policy_info.get("best_model_path"),
                        policy_info.get("fallback_reason"),
                    )
                    if policy_decision == "auto_hit" and policy_path.exists():
                        if CONFIG.enable_geometry_texture_split and not split_report_path(fused_keyword).exists():
                            split_model_for_cache(fused_keyword, policy_path)
                        push_event({"type": "status", "text": "策略缓存自动命中，已复用"})
                        push_event({
                            "type": "model",
                            "url": f"/models/model_cache/{policy_path.name}",
                            "keyword": fused_keyword,
                            "query_keyword": keyword,
                            "cache_hit": True,
                            "fused_cache_hit": True,
                            "policy_cache_hit": True,
                            "policy_score": policy_info.get("policy_score"),
                            **progressive_model_meta(fused_keyword, policy_path),
                        })
                        return
                    if policy_decision == "auto_hit" and not policy_path.exists():
                        logger.warning(
                            "策略缓存 auto_hit 但 best_model_path 不存在，按 miss 回退：%s",
                            policy_path,
                        )
                    elif policy_decision == "review":
                        candidate_model_url = ""
                        candidate_model_path = ""
                        if policy_path.exists():
                            candidate_model_url = f"/models/model_cache/{policy_path.name}"
                            candidate_model_path = str(policy_path)
                        review_event = {
                            "type": "cache_review",
                            "text": "发现相似缓存模型，后续可支持复用",
                            "candidate_model_url": candidate_model_url,
                            "candidate_model_path": candidate_model_path,
                            "policy_score": policy_info.get("policy_score"),
                            "text_score": policy_info.get("text_score"),
                            "image_score": policy_info.get("image_score"),
                            "best_keyword": fused_keyword,
                            "best_filename": str(fused_meta.get("best_filename") or policy_path.name),
                            "weak_threshold": policy_info.get("weak_threshold"),
                            "strong_threshold": policy_info.get("strong_threshold"),
                            "review_phase": "hint_only",
                            "fallback_reason": "" if policy_path.exists() else "best_model_path_missing",
                        }
                        push_event(review_event)
                        logger.info(
                            "策略缓存 review UI 提示已发送：review_ui_event_sent=True, "
                            "review_phase=hint_only, policy_decision=review, policy_score=%s, "
                            "best_model_path=%s, review_fallback_to_miss=True",
                            policy_info.get("policy_score"),
                            policy_info.get("best_model_path"),
                        )
                        logger.info(
                            "策略缓存 review：当前前端未接入 review，review 按 miss 回退原生成流程。"
                        )
                    else:
                        logger.info("策略缓存 miss，继续原生成流程。")
            else:
                fused_hit = find_fused_cached_model(
                    keyword=keyword,
                    explanation=explanation,
                    query_image=query_image_for_cache,
                    threshold=0.82,
                )
                if fused_hit:
                    fused_path, fused_score, fused_keyword = fused_hit
                    if CONFIG.enable_geometry_texture_split and not split_report_path(fused_keyword).exists():
                        split_model_for_cache(fused_keyword, fused_path)
                    logger.info(
                        "融合缓存命中：fused_score=%.4f, model_path=%s",
                        fused_score,
                        fused_path,
                    )
                    push_event({"type": "status", "text": "发现相似缓存模型，已复用"})
                    push_event({
                        "type": "model",
                        "url": f"/models/model_cache/{fused_path.name}",
                        "keyword": fused_keyword,
                        "query_keyword": keyword,
                        "cache_hit": True,
                        "fused_cache_hit": True,
                        "fused_score": round(fused_score, 4),
                        **progressive_model_meta(fused_keyword, fused_path),
                    })
                    return

            # 缓存未命中：本地 TripoSR 生成
            push_event({"type": "status", "text": f"本地生成 3D 模型：{keyword}"})
            model_path, error = generate_3d_model_local_triposr(roi_path_for_model, keyword)

            if error:
                push_event({"type": "error", "text": error})
                return

            if model_path:
                push_event({
                    "type": "model",
                    "url": f"/models/model_cache/{model_path.name}",
                    "keyword": keyword,
                    "cache_hit": False,
                    **progressive_model_meta(keyword, model_path),
                })
                push_event({"type": "status", "text": "本地 3D 模型已生成并缓存"})
            else:
                push_event({"type": "error", "text": "本地模型生成失败"})

        finally:
            processing_lock.release()

    threading.Thread(target=run, daemon=True).start()


# ==================== 手势识别辅助 ====================

class SwipeDetector:
    def __init__(self, maxlen: int = 8) -> None:
        self.points: Deque[Tuple[float, float, float]] = collections.deque(maxlen=maxlen)
        self.last_emit = 0.0

    def update(self, x: float, y: float) -> Optional[str]:
        now = time.time()
        self.points.append((now, x, y))
        if len(self.points) < self.points.maxlen:
            return None
        t0, x0, y0 = self.points[0]
        t1, x1, y1 = self.points[-1]
        dt = max(0.001, t1 - t0)
        dx = x1 - x0
        dy = y1 - y0
        if dt <= 0.75 and abs(dx) > 0.22 and abs(dy) < 0.14 and now - self.last_emit > 1.0:
            self.last_emit = now
            self.points.clear()
            return "swipe_right" if dx > 0 else "swipe_left"
        return None


def finger_extended(hand: List[Any], tip: int, pip: int, margin: float = 0.018) -> bool:
    return hand[tip].y < hand[pip].y - margin


def finger_down(hand: List[Any], tip: int, pip: int, margin: float = 0.018) -> bool:
    return hand[tip].y > hand[pip].y + margin


def classify_hand_gesture(hand: List[Any]) -> Tuple[str, str, float]:
    index_up = finger_extended(hand, 8, 6)
    middle_up = finger_extended(hand, 12, 10)
    ring_up = finger_extended(hand, 16, 14)
    pinky_up = finger_extended(hand, 20, 18)
    index_down = finger_down(hand, 8, 6)
    middle_down = finger_down(hand, 12, 10)
    ring_down = finger_down(hand, 16, 14)
    pinky_down = finger_down(hand, 20, 18)
    up_count = sum([index_up, middle_up, ring_up, pinky_up])
    down_count = sum([index_down, middle_down, ring_down, pinky_down])
    if up_count >= 4:
        return "open_palm", "张开手掌：待命", 0.88
    if up_count == 0 and down_count >= 3:
        return "fist", "握拳：确认/选中", 0.82
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point_up", "食指上指：向上滚动", 0.86
    if index_down and not middle_up and not ring_up and not pinky_up:
        return "point_down", "食指下指：向下滚动", 0.82
    if index_up and middle_up and not ring_up and not pinky_up:
        return "two_fingers_up", "两指上指：放大", 0.84
    if index_down and middle_down and not ring_up and not pinky_up:
        return "two_fingers_down", "两指下指：缩小", 0.80
    return "unknown", "未识别手势", 0.45


def draw_hand(frame: Any, hand: List[Any]) -> None:
    h, w, _ = frame.shape
    for a, b in HAND_LINES:
        p1 = hand[a]
        p2 = hand[b]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 192, 203), 2)
    for lm in hand:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 255), -1)


def get_roi_rect_from_points(points: List[Tuple[int, int]], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(points) < 4:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin = max(0, min(xs))
    ymin = max(0, min(ys))
    xmax = min(width - 1, max(xs))
    ymax = min(height - 1, max(ys))
    if xmax <= xmin or ymax <= ymin:
        return None
    area_ratio = ((xmax - xmin) * (ymax - ymin)) / max(1, width * height)
    return (xmin, ymin, xmax, ymax) if area_ratio >= CONFIG.min_roi_area_ratio else None


def is_rect_stable(prev: Optional[Tuple[int, int, int, int]], cur: Tuple[int, int, int, int], width: int, height: int) -> bool:
    if prev is None:
        return False
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = cur
    prev_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    cur_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
    center_shift = dist_px(prev_center, cur_center)
    size_shift = abs((px2 - px1) - (cx2 - cx1)) + abs((py2 - py1) - (cy2 - cy1))
    return center_shift < min(width, height) * 0.045 and size_shift < min(width, height) * 0.09


# ==================== 视频生成器 ====================

def generate_video():
    if not Path(CONFIG.model_path).exists():
        msg = "Hand model not found"
        logger.error("手势模型文件不存在：%s", CONFIG.model_path)
        push_event({"type": "error", "text": f"手势模型文件不存在：{CONFIG.model_path}"})
        jpeg = make_message_jpeg(msg)
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1)

    base_options = tasks.BaseOptions(model_asset_path=CONFIG.model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.IMAGE)
    hand_detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CONFIG.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)

    if not cap.isOpened():
        logger.error("摄像头打开失败：index=%s", CONFIG.camera_index)
        push_event({"type": "error", "text": "摄像头打开失败，请检查摄像头权限或 CAMERA_INDEX"})
        jpeg = make_message_jpeg("Camera error")
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1)

    stable_frames = 0
    cooldown_until = 0.0
    prev_dist = None
    prev_roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_frame_buffer: Deque[Any] = collections.deque(maxlen=max(1, CONFIG.roi_multiframe_count))
    last_gesture = ""
    last_gesture_ts = 0.0
    swipe = SwipeDetector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取摄像头帧失败")
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_detector.detect(mp_image)

            finger_points: List[Tuple[int, int]] = []
            active_hands: List[Tuple[float, float]] = []
            hands = hand_result.hand_landmarks or []

            if hands:
                gesture, detail, confidence = classify_hand_gesture(hands[0])
                swipe_gesture = swipe.update(hands[0][8].x, hands[0][8].y)
                if swipe_gesture:
                    gesture = swipe_gesture
                    detail = "右滑：旋转模型" if swipe_gesture == "swipe_right" else "左滑：旋转模型"
                    confidence = 0.90

                now = time.time()
                if gesture != last_gesture or now - last_gesture_ts >= CONFIG.gesture_event_interval:
                    push_event({"type": "gesture", "gesture": gesture, "detail": detail, "confidence": confidence})
                    last_gesture = gesture
                    last_gesture_ts = now

                for hand in hands:
                    draw_hand(frame, hand)
                    ix = int(hand[8].x * w)
                    iy = int(hand[8].y * h)
                    tx = int(hand[4].x * w)
                    ty = int(hand[4].y * h)
                    pinch_distance = dist_norm(hand[8], hand[4])
                    is_pinching = pinch_distance < CONFIG.pinch_threshold_norm
                    if is_pinching:
                        active_hands.append((hand[8].x, hand[8].y))
                        cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)
                        cv2.circle(frame, (tx, ty), 12, (0, 255, 0), -1)
                        cv2.line(frame, (ix, iy), (tx, ty), (255, 255, 255), 4)
                    else:
                        cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
                        cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)
                    finger_points.extend([(ix, iy), (tx, ty)])
            else:
                last_gesture = ""

            if len(active_hands) == 1:
                push_event({"type": "drag", "active": True, "x": active_hands[0][0], "y": active_hands[0][1]})
            elif active_hands:
                push_event({"type": "drag", "active": False, "x": 0.5, "y": 0.5})

            if len(active_hands) == 2:
                (x1, y1), (x2, y2) = active_hands
                cur = math.hypot(x2 - x1, y2 - y1)
                if prev_dist and prev_dist > 0.001:
                    factor = max(0.85, min(1.18, cur / prev_dist))
                    push_event({"type": "scale", "factor": factor})
                prev_dist = cur
            else:
                prev_dist = None

            roi_rect = None
            if len(hands) >= 2 and len(active_hands) == 0:
                roi_rect = get_roi_rect_from_points(finger_points, w, h)

            if roi_rect:
                if is_rect_stable(prev_roi_rect, roi_rect, w, h):
                    stable_frames += 1
                else:
                    stable_frames = 1
                prev_roi_rect = roi_rect
                xmin, ymin, xmax, ymax = roi_rect
                roi_frame_buffer.append(raw_frame[ymin:ymax, xmin:xmax].copy())
                ready = stable_frames >= CONFIG.trigger_frames
                color = (0, 255, 0) if ready else (0, 200, 200)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3 if ready else 2)
                cv2.putText(frame, f"Selecting {stable_frames}/{CONFIG.trigger_frames}", (xmin, max(30, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                stable_frames = 0
                prev_roi_rect = None
                roi_frame_buffer.clear()

            now = time.time()
            if now < cooldown_until:
                cv2.putText(frame, "Cooldown...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if roi_rect and stable_frames >= CONFIG.trigger_frames and now >= cooldown_until and not processing_lock.locked():
                if processing_lock.acquire(blocking=False):
                    stable_frames = 0
                    cooldown_until = now + CONFIG.cooldown_seconds
                    xmin, ymin, xmax, ymax = roi_rect
                    roi = raw_frame[ymin:ymax, xmin:xmax]
                    buffered_rois = list(roi_frame_buffer)
                    if CONFIG.enable_roi_keyframe_preprocess and len(buffered_rois) >= 2:
                        if len(buffered_rois) < CONFIG.roi_multiframe_count:
                            buffered_rois.append(roi.copy())
                        roi_path = save_roi_sequence(buffered_rois)
                    else:
                        roi_path = CONFIG.static_model_dir / f"selected_roi_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
                        cv2.imwrite(str(roi_path), roi)
                    roi_frame_buffer.clear()
                    logger.info("已保存 ROI 输入：%s", roi_path)
                    process_roi_async(roi_path)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG.jpeg_quality])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.01)
    finally:
        cap.release()
        try:
            hand_detector.close()
        except Exception:
            pass
        logger.info("视频流资源已释放")


# ==================== Flask 路由 ====================


CLOUD_REALTIME_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>??? AR 3D ????</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}html,body{width:100%;height:100%;background:#080b10;color:#f7fbff;font-family:Arial,"Microsoft YaHei",sans-serif;overflow:hidden}#app{position:relative;width:100vw;height:100vh;background:#0b0f14}.camera-wrap{position:absolute;inset:0;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}#camera{width:100%;height:100%;object-fit:cover;background:#000}@media(min-width:900px){#camera{width:auto;height:auto;max-width:min(100%,980px);max-height:min(100%,740px);border-radius:14px;box-shadow:0 18px 50px rgba(0,0,0,.28);object-fit:contain}}#roi{position:absolute;left:50%;top:50%;width:min(46vw,360px);height:min(32vw,240px);transform:translate(-50%,-50%);border:3px solid #4fe0ff;border-radius:16px;box-shadow:0 0 0 9999px rgba(0,0,0,.18),0 0 22px rgba(79,224,255,.4);touch-action:none;cursor:move;z-index:5}.corner{position:absolute;width:18px;height:18px;border:3px solid #fff;background:#4fe0ff;border-radius:50%;right:-10px;bottom:-10px;cursor:nwse-resize}.topbar{position:absolute;left:50%;top:14px;transform:translateX(-50%);z-index:20;max-width:94vw;background:rgba(2,8,14,.72);border:1px solid rgba(255,255,255,.14);backdrop-filter:blur(8px);padding:9px 16px;border-radius:999px;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.panel{position:absolute;right:16px;top:62px;z-index:20;width:min(360px,calc(100vw - 32px));background:rgba(5,12,18,.78);border:1px solid rgba(255,255,255,.16);border-radius:16px;padding:14px;backdrop-filter:blur(10px);box-shadow:0 12px 38px rgba(0,0,0,.28)}.panel h2{font-size:17px;margin-bottom:8px}.muted{font-size:12px;color:rgba(255,255,255,.72);line-height:1.45}.chips{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:12px 0}.chips button,.primary,.ghost{border:none;border-radius:10px;padding:10px 12px;font-size:14px;cursor:pointer;color:#fff;background:rgba(255,255,255,.13)}.chips button.active{background:#2b9fff;box-shadow:0 0 0 1px rgba(255,255,255,.25) inset}.primary{width:100%;background:#16b98e;font-weight:bold;margin-top:6px}.ghost{width:100%;background:rgba(255,255,255,.10);margin-top:8px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.metric{background:rgba(255,255,255,.08);border-radius:10px;padding:8px}.metric label{display:block;font-size:11px;color:rgba(255,255,255,.62)}.metric strong{font-size:16px}.decision{margin-top:12px;padding:10px;border-radius:12px;background:rgba(79,224,255,.10);border:1px solid rgba(79,224,255,.25);font-size:13px;line-height:1.5}.model-layer{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);z-index:15;width:min(360px,86vw);height:min(320px,46vh);display:none;background:rgba(0,0,0,.24);border:1px solid rgba(255,255,255,.12);border-radius:18px;overflow:hidden;backdrop-filter:blur(4px)}.model-layer.show{display:block}.model-layer model-viewer{width:100%;height:100%;background:transparent}.review{position:absolute;left:16px;top:62px;z-index:21;width:min(340px,calc(100vw - 32px));display:none;background:rgba(5,12,18,.82);border:1px solid rgba(255,208,87,.32);border-radius:14px;padding:13px;backdrop-filter:blur(10px)}.review.show{display:block}.review h3{font-size:16px;margin-bottom:6px}.review p{font-size:13px;color:rgba(255,255,255,.82);line-height:1.5}.thumb{position:absolute;left:16px;bottom:18px;z-index:22;width:150px;height:100px;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.22);background:rgba(0,0,0,.5);display:none}.thumb.show{display:block}.thumb canvas{width:100%;height:100%;object-fit:cover}.hint{position:absolute;left:16px;top:14px;z-index:20;background:rgba(2,8,14,.58);border:1px solid rgba(255,255,255,.12);padding:8px 12px;border-radius:999px;font-size:12px;color:rgba(255,255,255,.78)}@media(max-width:720px){.panel{left:12px;right:12px;top:auto;bottom:12px;width:auto}.model-layer{bottom:260px;height:240px}.review{top:58px;left:12px;right:12px;width:auto}.thumb{display:none!important}.chips{grid-template-columns:repeat(3,1fr)}.chips button{padding:8px;font-size:12px}.topbar{font-size:12px}.hint{display:none}}
</style>
</head>
<body>
<div id="app"><div class="camera-wrap"><video id="camera" autoplay playsinline muted></video></div><div id="roi"><div class="corner"></div></div><div class="hint">?????? ROI???????? 3D ??</div><div id="status" class="topbar">????????????...</div><div id="review" class="review"><h3>??????</h3><p id="reviewText">???????????????????????????</p></div><div id="modelLayer" class="model-layer"><model-viewer id="viewer" camera-controls auto-rotate ar exposure="1" shadow-intensity="0.7"></model-viewer></div><div id="thumb" class="thumb"><canvas id="roiCanvas" width="320" height="220"></canvas></div><div class="panel"><h2>AR 3D ????</h2><div class="muted">?????????????????? ROI ???????????????????????????????</div><div class="chips" id="chips"><button data-kind="glasses" class="active">??</button><button data-kind="cup">??</button><button data-kind="box">???</button><button data-kind="camera">??</button><button data-kind="helmet">??</button><button data-kind="other">??</button></div><button id="runBtn" class="primary">????? 3D</button><button id="resetBtn" class="ghost">????</button><div class="metrics"><div class="metric"><label>policy_score</label><strong id="score">--</strong></div><div class="metric"><label>decision</label><strong id="decision">--</strong></div><div class="metric"><label>strong</label><strong>0.74</strong></div><div class="metric"><label>weak</label><strong>0.60</strong></div></div><div id="decisionText" class="decision">???? ROI?</div></div></div>
<script>
const camera=document.getElementById('camera'),statusEl=document.getElementById('status'),roi=document.getElementById('roi'),chips=document.getElementById('chips'),runBtn=document.getElementById('runBtn'),resetBtn=document.getElementById('resetBtn'),scoreEl=document.getElementById('score'),decisionEl=document.getElementById('decision'),decisionText=document.getElementById('decisionText'),modelLayer=document.getElementById('modelLayer'),viewer=document.getElementById('viewer'),review=document.getElementById('review'),reviewText=document.getElementById('reviewText'),thumb=document.getElementById('thumb'),roiCanvas=document.getElementById('roiCanvas');
const assets={glasses:{label:'??',score:.92,decision:'auto_hit',model:'/models/competition_demo_models_hq/sunglasses_hq.glb',note:'?????????????? 3D ???'},cup:{label:'??/??',score:.86,decision:'auto_hit',model:'/models/competition_demo_models_hq/water_bottle_hq.glb',note:'?????????????? GLB?'},box:{label:'???',score:.67,decision:'low_confidence_candidate',model:'/models/competition_demo_models_hq/boombox_hq.glb',note:'?????????????????????????????'},camera:{label:'??',score:.88,decision:'auto_hit',model:'/models/competition_demo_models_hq/antique_camera_hq.glb',note:'?????????????????'},helmet:{label:'??',score:.82,decision:'auto_hit',model:'/models/competition_demo_models_hq/damaged_helmet_hq.glb',note:'?????????????????'},other:{label:'????',score:.31,decision:'miss',model:'',note:'???????????????????'}};
let selected='glasses',drag=false,resize=false,startX=0,startY=0,startRect=null;
async function startCamera(){try{const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});camera.srcObject=stream;statusEl.textContent='??????????????? ROI ?';}catch(err){statusEl.textContent='???????????????????';decisionText.textContent='???????????????? ROI?';console.error(err)}}
function setStatus(t){statusEl.textContent=t}function activeChip(kind){selected=kind;document.querySelectorAll('#chips button').forEach(b=>b.classList.toggle('active',b.dataset.kind===kind))}chips.addEventListener('click',e=>{if(e.target.dataset.kind)activeChip(e.target.dataset.kind)});
function rectInfo(){const r=roi.getBoundingClientRect(),v=camera.getBoundingClientRect();return{x:r.left-v.left,y:r.top-v.top,w:r.width,h:r.height}}function captureRoi(){const ctx=roiCanvas.getContext('2d'),info=rectInfo();roiCanvas.width=Math.max(1,Math.round(info.w));roiCanvas.height=Math.max(1,Math.round(info.h));try{ctx.drawImage(camera,info.x,info.y,info.w,info.h,0,0,roiCanvas.width,roiCanvas.height);thumb.classList.add('show')}catch(e){console.warn('ROI capture skipped',e)}}
function showResult(kind){const a=assets[kind];captureRoi();scoreEl.textContent=a.score.toFixed(2);decisionEl.textContent=a.decision;review.classList.remove('show');modelLayer.classList.remove('show');if(a.decision==='auto_hit'){viewer.src=a.model+'?t='+Date.now();modelLayer.classList.add('show');decisionText.textContent=`${a.label}?${a.note} baseline ??? 51 ??????????????`;setStatus('???????? 3D ??');}else if(a.decision==='low_confidence_candidate'){viewer.src=a.model+'?t='+Date.now();modelLayer.classList.add('show');reviewText.textContent=`${a.label}?policy_score=${a.score.toFixed(2)}??? 0.60~0.74 ????????`;review.classList.add('show');decisionText.textContent=a.note;setStatus('?????????????????');}else{viewer.removeAttribute('src');decisionText.textContent=`${a.label}?${a.note}`;setStatus('???????????');}}
runBtn.addEventListener('click',()=>showResult(selected));resetBtn.addEventListener('click',()=>{modelLayer.classList.remove('show');review.classList.remove('show');thumb.classList.remove('show');scoreEl.textContent='--';decisionEl.textContent='--';decisionText.textContent='???? ROI?';setStatus('??????????????? ROI ?')});
function begin(e,mode){const p=e.touches?e.touches[0]:e;drag=mode==='drag';resize=mode==='resize';startX=p.clientX;startY=p.clientY;startRect=roi.getBoundingClientRect();e.preventDefault()}roi.addEventListener('mousedown',e=>{if(e.target.classList.contains('corner'))begin(e,'resize');else begin(e,'drag')});roi.addEventListener('touchstart',e=>{if(e.target.classList.contains('corner'))begin(e,'resize');else begin(e,'drag')},{passive:false});window.addEventListener('mousemove',move);window.addEventListener('touchmove',move,{passive:false});function move(e){if(!drag&&!resize)return;const p=e.touches?e.touches[0]:e,dx=p.clientX-startX,dy=p.clientY-startY;if(drag){roi.style.left=(startRect.left+startRect.width/2+dx)+'px';roi.style.top=(startRect.top+startRect.height/2+dy)+'px';roi.style.transform='translate(-50%,-50%)'}else{roi.style.width=Math.max(120,startRect.width+dx)+'px';roi.style.height=Math.max(90,startRect.height+dy)+'px'}e.preventDefault()}window.addEventListener('mouseup',()=>{drag=false;resize=false});window.addEventListener('touchend',()=>{drag=false;resize=false});
startCamera();console.log('browser-camera TIFS-Cache demo ready');
</script>
</body>
</html>"""

@app.route("/")
def index():
    cloud_demo_path = CONFIG.static_model_dir / "cloud_realtime_demo.html"
    if cloud_demo_path.exists():
        response = Response(cloud_demo_path.read_text(encoding="utf-8"), mimetype="text/html; charset=utf-8")
    else:
        response = Response(CLOUD_REALTIME_HTML, mimetype="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/video_feed")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/models/<path:filename>")
def serve_model(filename: str):
    return send_from_directory(str(CONFIG.static_model_dir.resolve()), filename)


@app.route("/events")
def sse():
    def stream():
        while True:
            try:
                msg = sse_queue.get(timeout=0.5)
                yield sse_pack(msg)
            except queue.Empty:
                yield ": keep-alive\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return Response(stream(), mimetype="text/event-stream", headers=headers)


@app.route("/health")
def health():
    cache_index = load_cache_index()
    return {
        "ok": True,
        "model_exists": Path(CONFIG.model_path).exists(),
        "model_path": CONFIG.model_path,
        "static_model_dir": str(CONFIG.static_model_dir.resolve()),
        "model_cache_dir": str(MODEL_CACHE_DIR.resolve()),
        "cache_count": len(cache_index),
        "local_vlm_model": CONFIG.local_vlm_model,
        "triposr_dir": CONFIG.triposr_dir,
        "triposr_configured": bool(CONFIG.triposr_dir and (Path(CONFIG.triposr_dir) / "run.py").exists()),
        "geometry_texture_split_enabled": CONFIG.enable_geometry_texture_split,
        "progressive_model_loading_enabled": CONFIG.enable_progressive_model_loading,
        "roi_keyframe_preprocess_enabled": CONFIG.enable_roi_keyframe_preprocess,
        "roi_multiframe_count": CONFIG.roi_multiframe_count,
        "model_split_dir": str(MODEL_SPLIT_DIR.resolve()),
        "processing": processing_lock.locked(),
    }


@app.route("/cache")
def cache_status():
    return {"cache_dir": str(MODEL_CACHE_DIR.resolve()), "items": load_cache_index()}


# ==================== 程序入口 ====================

if __name__ == "__main__":
    print("=" * 74)
    print("   🚀 AR 手势阅读助手 - 开源模型版 v3.6.1")
    print("   http://localhost:5000")
    print("   图像理解：Qwen2.5-VL 本地开源模型")
    print("   3D生成：TripoSR 本地开源 image-to-3D")
    print("   缓存目录：runtime_assets/model_cache")
    print("   必填：HAND_MODEL_PATH / TRIPOSR_DIR / TRIPOSR_PYTHON")
    print("=" * 74)

    if not CONFIG.triposr_dir:
        logger.warning("尚未配置 TRIPOSR_DIR。识别可以运行，但 3D 生成会失败。")
    elif not (Path(CONFIG.triposr_dir) / "run.py").exists():
        logger.warning("TRIPOSR_DIR 中未找到 run.py：%s", CONFIG.triposr_dir)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

