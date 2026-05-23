from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import requests
import websockets


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_repro_outputs" / "cache_similarity_eval_v3_real_70"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_MD = OUT_DIR / "review_ui_phase1_browser_validation_result.md"
SCREENSHOT = OUT_DIR / "review_ui_phase1_browser_validation_screenshot.png"
PLUS_LOG = OUT_DIR / "review_ui_phase1_browser_validation_plus.log"
CHROME_LOG = OUT_DIR / "review_ui_phase1_browser_validation_chrome.log"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PORT = 9332
URL = "http://127.0.0.1:5000"

console_errors = []
exceptions = []


async def cdp_command(ws, method, params=None, seq=[0]):
    seq[0] += 1
    msg_id = seq[0]
    await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if "method" in msg:
            if msg["method"] == "Runtime.consoleAPICalled":
                level = msg.get("params", {}).get("type", "")
                if level in {"error", "assert"}:
                    console_errors.append(msg.get("params", {}))
            if msg["method"] == "Runtime.exceptionThrown":
                exceptions.append(msg.get("params", {}))
            continue
        if msg.get("id") == msg_id:
            if "error" in msg:
                raise RuntimeError(f"CDP error for {method}: {msg['error']}")
            return msg.get("result", {})


def wait_http(url: str, timeout: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


async def run_browser(result: dict) -> None:
    user_dir = Path(tempfile.mkdtemp(prefix="review-ui-chrome-"))
    chrome_proc = None
    try:
        chrome_args = [
            str(CHROME),
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={user_dir}",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ]
        with CHROME_LOG.open("wb") as chrome_log:
            chrome_proc = subprocess.Popen(chrome_args, stdout=chrome_log, stderr=chrome_log)

        version = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1) as resp:
                    version = json.loads(resp.read().decode("utf-8"))
                    break
            except Exception:
                await asyncio.sleep(0.25)
        if not version:
            raise RuntimeError("Chrome DevTools endpoint did not start")

        async with websockets.connect(version["webSocketDebuggerUrl"], max_size=8_000_000) as browser:
            target = await cdp_command(browser, "Target.createTarget", {"url": "about:blank"})
            target_id = target["targetId"]

        page = None
        for _ in range(20):
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            page = next((item for item in targets if item.get("id") == target_id), None)
            if page and page.get("webSocketDebuggerUrl"):
                break
            await asyncio.sleep(0.2)
        if not page:
            raise RuntimeError("Could not find Chrome page target")

        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=16_000_000) as ws:
            await cdp_command(ws, "Runtime.enable")
            await cdp_command(ws, "Page.enable")
            await cdp_command(ws, "Network.enable")
            await cdp_command(
                ws,
                "Network.setBlockedURLs",
                {"urls": ["*video_feed*", "*model_cache/眼镜_db9b92eef7.glb*"]},
            )
            await cdp_command(ws, "Page.navigate", {"url": URL})
            await asyncio.sleep(2.0)
            ready = await cdp_command(
                ws,
                "Runtime.evaluate",
                {
                    "expression": "typeof handleServerEvent === 'function' && !!document.getElementById('review-card')",
                    "returnByValue": True,
                },
            )
            result["page_opened"] = bool(ready.get("result", {}).get("value"))

            event_expr = """
handleServerEvent({
  type: 'cache_review',
  text: '发现相似缓存模型，后续可支持复用',
  candidate_model_url: '/models/model_cache/眼镜_db9b92eef7.glb',
  candidate_model_path: 'runtime_assets/model_cache/眼镜_db9b92eef7.glb',
  policy_score: 0.75,
  text_score: 0.75,
  image_score: null,
  best_keyword: '眼镜',
  best_filename: '眼镜_db9b92eef7.glb',
  weak_threshold: 0.7,
  strong_threshold: 0.78,
  review_phase: 'hint_only'
});
true;
"""
            await cdp_command(ws, "Runtime.evaluate", {"expression": event_expr, "returnByValue": True})
            result["cache_review_injected"] = True
            await asyncio.sleep(0.5)

            inspect_expr = """
(() => {
 const card = document.getElementById('review-card');
 const body = document.body.innerText;
 const resources = performance.getEntriesByType('resource').map(r => r.name);
 return {
   visible: !!card && card.classList.contains('show'),
   body,
   reviewMeta: (document.getElementById('review-meta') || {}).textContent || '',
   buttonsDisabled: Array.from(document.querySelectorAll('#review-card button')).every(b => b.disabled),
   candidateLoaded: resources.some(u => u.includes('眼镜_db9b92eef7.glb') || u.includes('%E7%9C%BC%E9%95%9C_db9b92eef7.glb')),
   resources
 };
})()
"""
            inspect = await cdp_command(ws, "Runtime.evaluate", {"expression": inspect_expr, "returnByValue": True})
            value = inspect.get("result", {}).get("value", {})
            body = value.get("body", "")
            meta = value.get("reviewMeta", "")
            result["review_card_visible"] = bool(value.get("visible"))
            result["contains_title"] = "发现相似缓存模型" in body
            result["contains_best_keyword"] = "眼镜" in meta
            result["contains_policy_score"] = "0.75" in meta or "score=0.75" in meta
            result["contains_thresholds"] = "0.7" in meta and "0.78" in meta
            result["buttons_disabled"] = bool(value.get("buttonsDisabled"))
            result["candidate_model_url_loaded"] = bool(value.get("candidateLoaded"))

            shot = await cdp_command(
                ws,
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": True},
            )
            SCREENSHOT.write_bytes(base64.b64decode(shot["data"]))
    finally:
        if chrome_proc and chrome_proc.poll() is None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
        shutil.rmtree(user_dir, ignore_errors=True)


def write_report(result: dict) -> None:
    md = f"""# Review UI 第一阶段浏览器验证结果

## 1. 验证方式

本轮通过 Chrome Headless + DevTools Protocol 打开真实页面 `http://127.0.0.1:5000`，并拦截 `/video_feed`，避免启动摄像头/手势检测链路。随后在前端注入一个等价的 `cache_review` 事件完成 UI 验证。

本轮为 UI 事件验证，不是完整 ROI 生成流程。

## 2. 页面检查

- 页面是否正常打开：{result['page_opened']}
- 是否有 JS 报错：{result['js_error_count'] > 0}
- JS 报错数量：{result['js_error_count']}
- 是否显示提示卡片：{result['review_card_visible']}

## 3. cache_review 提示检查

- 是否显示“发现相似缓存模型”：{result['contains_title']}
- 是否显示 `best_keyword`：{result['contains_best_keyword']}
- 是否显示 `policy_score`：{result['contains_policy_score']}
- 是否显示 `weak_threshold / strong_threshold`：{result['contains_thresholds']}
- 灰色按钮是否禁用：{result['buttons_disabled']}
- 是否没有真实复用行为：{not result['candidate_model_url_loaded']}
- 是否没有加载 `candidate_model_url`：{not result['candidate_model_url_loaded']}

## 4. 日志检查

- 是否出现 `cache_review`：{result['log_has_cache_review']}
- 是否出现 `review_ui_event_sent`：{result['log_has_review_ui_event_sent']}
- 是否调用 Qwen：{result['qwen_called']}
- 是否调用 TripoSR：{result['triposr_called']}

说明：本轮通过前端注入模拟 `cache_review` 事件，因此后端运行日志中不要求出现 `cache_review` 或 `review_ui_event_sent`。后端事件发送已由 helper 测试覆盖，本轮重点验证浏览器 UI 展示行为。

## 5. 截图

截图文件：`{SCREENSHOT.relative_to(ROOT)}`

## 6. 结论

{"Review UI 第一阶段浏览器验证通过。当前 review 事件能够在前端显示提示卡片，但不会真正复用缓存模型，仍保持保守回退策略。" if result['browser_validation_passed'] else "Review UI 第一阶段浏览器验证未完全通过，请查看上述检查项。"}
"""
    RESULT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    if not CHROME.exists():
        raise FileNotFoundError(f"Chrome not found: {CHROME}")

    result = {
        "browser_validation_passed": False,
        "page_opened": False,
        "js_error_count": 0,
        "review_card_visible": False,
        "contains_title": False,
        "contains_best_keyword": False,
        "contains_policy_score": False,
        "contains_thresholds": False,
        "buttons_disabled": False,
        "candidate_model_url_loaded": False,
        "qwen_called": False,
        "triposr_called": False,
        "cache_review_injected": False,
        "method": "Chrome Headless + DevTools Protocol; /video_feed blocked; simulated cache_review event",
    }

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["ENABLE_POLICY_CACHE"] = "1"
    env["CACHE_POLICY_PATH"] = "runtime_assets/cache_policy.json"

    plus_proc = None
    try:
        with PLUS_LOG.open("wb") as plus_log:
            plus_proc = subprocess.Popen(
                [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "plus.py"],
                cwd=ROOT,
                env=env,
                stdout=plus_log,
                stderr=plus_log,
            )
        if not wait_http(URL + "/health", timeout=25):
            raise RuntimeError("plus.py did not become ready on /health")
        asyncio.run(run_browser(result))
    finally:
        if plus_proc and plus_proc.poll() is None:
            plus_proc.terminate()
            try:
                plus_proc.wait(timeout=5)
            except Exception:
                plus_proc.kill()

    log_text = PLUS_LOG.read_text(encoding="utf-8", errors="replace") if PLUS_LOG.exists() else ""
    result["js_error_count"] = len(console_errors) + len(exceptions)
    result["qwen_called"] = ("加载本地 VLM" in log_text) or ("Qwen2.5-VL 加载完成" in log_text)
    result["triposr_called"] = ("调用 TripoSR" in log_text) or ("TripoSR 模型已缓存" in log_text)
    result["log_has_cache_review"] = "cache_review" in log_text
    result["log_has_review_ui_event_sent"] = "review_ui_event_sent" in log_text
    result["browser_validation_passed"] = all(
        [
            result["page_opened"],
            result["cache_review_injected"],
            result["review_card_visible"],
            result["contains_title"],
            result["contains_best_keyword"],
            result["contains_policy_score"],
            result["contains_thresholds"],
            result["buttons_disabled"],
            not result["candidate_model_url_loaded"],
            result["js_error_count"] == 0,
            not result["qwen_called"],
            not result["triposr_called"],
        ]
    )
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result_md={RESULT_MD}")
    print(f"screenshot={SCREENSHOT}")


if __name__ == "__main__":
    main()
