import asyncio, os, re, json, subprocess
from collections import deque
from typing import Deque, List, Tuple, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ================ 固定路径与参数（按需改） ================
TESTER_PATH   = r"C:\gddr7_tester\gddr7_tester.exe"
FRACTION      = 0.80          # 默认占用比例
CHUNK_ITERS   = 100           # 每组迭代
RING_SIZE     = 800           # 前端保留多少个点
ROLLING_WINDOW = 60           # 计算σ的滚动窗口长度（每个Mode各自维护）

# Afterburner Profile 模式：滑块 0..4 -> Profile1..5
PROFILE_MODE = True
AFTERBURNER_EXE = r"C:\Program Files (x86)\MSI Afterburner\MSIAfterburner.exe"
PROFILE_MAP = {0:1, 1:2, 2:3, 3:4, 4:5}

# 如果改用命令模板方式，把 PROFILE_MODE 设 False，并设置下面模板
SET_OFFSET_CMD_TEMPLATE: Optional[str] = None
# 例：SET_OFFSET_CMD_TEMPLATE = r'"C:\Tools\nvidiaInspector.exe" -setMemoryClockOffset:0,0,{offset}'

SLIDER_MIN  = 0
SLIDER_MAX  = 4 if PROFILE_MODE else 2000
SLIDER_STEP = 1 if PROFILE_MODE else 50
# ========================================================

app = FastAPI()
clients: List[WebSocket] = []
proc: Optional[asyncio.subprocess.Process] = None
RUN_ENABLED = True  # 是否运行测试器（Stop/Start 会改它）

# 解析 tester 输出（含 Mode）
LINE_RE = re.compile(
    r"\[Chunk\s+(\d+)\s*\|\s*Mode\s+(\d+)\]\s+Time:\s+([\d\.]+)\s+ms\s+\|\s+Bandwidth:\s+([\d\.]+)\s+GB/s\s+\|\s+New errors:\s+(\d+)\s+\|\s+Total errors:\s+(\d+)"
)

Point = Tuple[int,int,float,float,int,int]  # (chunk, mode, ms, gbps, new_err, total_err)
history: Deque[Point] = deque(maxlen=RING_SIZE)

# 每个 Mode 独立窗口，计算“组内抖动”
mode_hist = {m: deque(maxlen=ROLLING_WINDOW) for m in range(1, 6)}

def std_of(seq):
    n = len(seq)
    if n <= 1: return 0.0
    mu = sum(seq) / n
    return (sum((x - mu) ** 2 for x in seq) / n) ** 0.5

async def broadcast(msg: dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for d in dead:
        try: clients.remove(d)
        except ValueError: pass

async def runner_loop():
    """根据 RUN_ENABLED 状态管理 tester 进程，并把输出行推给前端"""
    global proc, history, mode_hist
    while True:
        if not RUN_ENABLED:
            await asyncio.sleep(0.2)
            continue

        cmd = [TESTER_PATH, f"{FRACTION:.2f}", f"{CHUNK_ITERS}"]
        if not os.path.isfile(TESTER_PATH):
            await broadcast({"type":"status","error":f"Tester not found: {TESTER_PATH}"})
            await asyncio.sleep(2); continue

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            await broadcast({"type":"status","text":f"tester started: {' '.join(cmd)}"})
        except Exception as e:
            await broadcast({"type":"status","error":f"failed to start tester: {e}"})
            await asyncio.sleep(2); continue

        try:
            while RUN_ENABLED:
                line = await proc.stdout.readline()
                if not line: break
                text = line.decode(errors="ignore").strip()
                m = LINE_RE.search(text)
                if m:
                    chunk = int(m.group(1))
                    mode  = int(m.group(2))
                    ms    = float(m.group(3))
                    gbps  = float(m.group(4))
                    newe  = int(m.group(5))
                    tote  = int(m.group(6))

                    # 组内窗口 + 统计
                    buf = mode_hist.get(mode)
                    if buf is not None:
                        buf.append(gbps)
                    per_mode_std = {str(k): std_of(v) for k, v in mode_hist.items() if len(v) > 1}
                    avg_std = (sum(per_mode_std.values()) / len(per_mode_std)) if per_mode_std else 0.0

                    # 保存历史点
                    history.append((chunk, mode, ms, gbps, newe, tote))

                    # 推送给前端
                    await broadcast({
                        "type":"point",
                        "chunk":chunk, "mode":mode, "ms":ms, "gbps":gbps,
                        "new_errors":newe, "total_errors":tote,
                        "per_mode_std": per_mode_std, "avg_std": avg_std
                    })
                elif "CUDA error" in text:
                    await broadcast({"type":"status","error":text})
        finally:
            # 结束当前进程
            if proc and proc.returncode is None:
                try: proc.terminate()
                except ProcessLookupError: pass
            try:
                code = await proc.wait()
            except Exception:
                code = -1
            await broadcast({"type":"status","text":f"tester exited with code {code}"})
            proc = None
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def _startup():
    asyncio.create_task(runner_loop())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    # 初始回放历史点（σ用当前窗口的统计）
    per_mode_std = {str(k): std_of(v) for k, v in mode_hist.items() if len(v) > 1}
    avg_std = (sum(per_mode_std.values()) / len(per_mode_std)) if per_mode_std else 0.0
    for (chunk, mode, ms, gbps, newe, tote) in list(history):
        await ws.send_text(json.dumps({
            "type":"point","chunk":chunk,"mode":mode,"ms":ms,"gbps":gbps,
            "new_errors":newe,"total_errors":tote,
            "per_mode_std": per_mode_std, "avg_std": avg_std
        }))
    try:
        while True: _ = await ws.receive_text()
    except WebSocketDisconnect:
        try: clients.remove(ws)
        except ValueError: pass

# -------- 控制接口：Start / Stop / Restart / SetMem --------

@app.post("/api/start")
async def api_start(req: Request):
    """开始：若已在跑则不动；可携带 fraction/iters 覆盖"""
    global RUN_ENABLED, FRACTION, CHUNK_ITERS
    data = await req.json()
    try:
        if "fraction" in data: FRACTION = float(data["fraction"])
        if "iters" in data: CHUNK_ITERS = int(data["iters"])
    except Exception:
        pass
    RUN_ENABLED = True
    return JSONResponse({"ok": True, "msg": "tester start requested", "running": RUN_ENABLED,
                         "fraction":FRACTION, "iters":CHUNK_ITERS})

@app.post("/api/stop")
async def api_stop():
    """停止：终止当前进程，并禁止自动重启"""
    global RUN_ENABLED, proc
    RUN_ENABLED = False
    if proc and proc.returncode is None:
        try: proc.terminate()
        except ProcessLookupError: pass
    return JSONResponse({"ok": True, "msg": "tester stop requested"})

@app.post("/api/restart")
async def api_restart(req: Request):
    """重启：清空历史与窗口，立即重启"""
    global FRACTION, CHUNK_ITERS, proc, history, mode_hist, RUN_ENABLED
    data = await req.json()
    try:
        FRACTION = float(data.get("fraction", FRACTION))
        CHUNK_ITERS = int(data.get("iters", CHUNK_ITERS))
    except Exception:
        pass
    history.clear()
    mode_hist = {m: deque(maxlen=ROLLING_WINDOW) for m in range(1, 6)}
    RUN_ENABLED = True
    if proc and proc.returncode is None:
        try: proc.terminate()
        except ProcessLookupError: pass
    return JSONResponse({"ok":True,"msg":"restarting tester","fraction":FRACTION,"iters":CHUNK_ITERS})

def run_cmd(cmd: str):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return False, str(e)

@app.post("/api/set_mem")
async def api_set_mem(req: Request):
    """应用 Profile / 偏移；成功后清空统计窗口（让新频率的σ立即独立）"""
    global mode_hist, history
    if PROFILE_MODE:
        data = await req.json()
        level = int(data.get("level", 0))
        prof = PROFILE_MAP.get(level)
        if prof is None:
            return JSONResponse({"ok":False,"msg":f"level {level} not mapped"}, status_code=400)
        if not os.path.isfile(AFTERBURNER_EXE):
            return JSONResponse({"ok":False,"msg":f"Afterburner not found: {AFTERBURNER_EXE}"}, status_code=400)
        cmd = f'"{AFTERBURNER_EXE}" -Profile{prof}'
        ok, out = run_cmd(cmd)
        if ok:
            mode_hist = {m: deque(maxlen=ROLLING_WINDOW) for m in range(1, 6)}  # 清零σ窗口
            await broadcast({"type":"status","text":f"Profile{prof} applied; stats cleared"})
        return JSONResponse({"ok":ok,"cmd":cmd,"out":out})
    else:
        if not SET_OFFSET_CMD_TEMPLATE:
            return JSONResponse({"ok":False,"msg":"SET_OFFSET_CMD_TEMPLATE not configured"}, status_code=400)
        data = await req.json()
        offset = int(data.get("offset", 0))
        cmd = SET_OFFSET_CMD_TEMPLATE.format(offset=offset)
        ok, out = run_cmd(cmd)
        if ok:
            mode_hist = {m: deque(maxlen=ROLLING_WINDOW) for m in range(1, 6)}
            await broadcast({"type":"status","text":f"Offset {offset} applied; stats cleared"})
        return JSONResponse({"ok":ok,"cmd":cmd,"out":out})

# ------------------- 纯字符串 HTML（避免 f-string 与 {{}} 冲突） -------------------
INDEX_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"/><title>GDDR7 Live Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
 body {font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:20px}
 .row {display:flex;gap:16px;flex-wrap:wrap}
 .card {flex:1 1 520px;padding:16px;border:1px solid #ddd;border-radius:12px;box-shadow:0 1px 6px rgba(0,0,0,.05)}
 .muted {color:#666;font-size:13px}
 .label {font-weight:600}
 input[type=range] {width:100%}
 button {padding:8px 12px;border-radius:8px;border:1px solid #ccc;background:#f8f8f8;cursor:pointer}
 button:hover {background:#f0f0f0}
 .legend span {display:inline-block;margin-right:10px}
 .dot {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
 .pill {display:inline-block;padding:6px 10px;border:1px solid #ccc;border-radius:999px;margin-right:8px;background:#fafafa;cursor:pointer}
 .pill:hover {background:#f0f0f0}
</style>
</head><body>
<h2>GDDR7 Live Monitor</h2>
<script>const CFG = __CFG_JSON__;</script>
<div class="row">
  <div class="card">
    <canvas id="bwChart" height="260"></canvas>
    <div class="legend muted" style="margin-top:6px">
      <span><i class="dot" style="background:#1e88e5"></i>Mode1 linear</span>
      <span><i class="dot" style="background:#43a047"></i>Mode2 stride 64KiB</span>
      <span><i class="dot" style="background:#f4511e"></i>Mode3 stride 128KiB</span>
      <span><i class="dot" style="background:#8e24aa"></i>Mode4 block-xor</span>
      <span><i class="dot" style="background:#fdd835"></i>Mode5 permute</span>
    </div>
  </div>
  <div class="card" style="max-width:420px">
    <div style="margin-bottom:8px">
      <span class="pill" id="btnStart">开始</span>
      <span class="pill" id="btnStop">停止</span>
      <span class="pill" id="btnRestart">重启</span>
      <span id="status" class="muted" style="margin-left:8px"></span>
    </div>
    <div><span class="label">当前带宽：</span><span id="bw">--</span> GB/s</div>
    <div><span class="label">总错误：</span><span id="errs">0</span></div>
    <div style="margin-top:6px">
      <span class="label">抖动(σ)：</span>
      <div class="muted" id="sigmaTable">
        <div>Mode1: <span id="s1">--</span> | Mode2: <span id="s2">--</span> | Mode3: <span id="s3">--</span></div>
        <div>Mode4: <span id="s4">--</span> | Mode5: <span id="s5">--</span> | 平均σ̄: <span id="savg">--</span></div>
      </div>
    </div>
    <hr/>
    <div><span class="label">占用比例</span> <input id="fraction" type="number" min="0.1" max="0.9" step="0.05"></div>
    <div style="margin-top:8px"><span class="label">每组迭代</span> <input id="iters" type="number" min="10" max="1000" step="10"></div>
    <hr/>
    <div><span class="label">显存频率滑块</span>（需 Afterburner）</div>
    <input id="memSlider" type="range">
    <div class="muted" id="sliderLabel"></div>
    <div style="margin-top:8px"><button id="btnApply">应用</button> <span id="applyMsg" class="muted"></span></div>
  </div>
</div>
<script>
const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
const labels=[], bw=[], modes=[], colors=[];
const palette={1:'#1e88e5',2:'#43a047',3:'#f4511e',4:'#8e24aa',5:'#fdd835'};

const ctx=document.getElementById('bwChart').getContext('2d');
const chart=new Chart(ctx,{type:'line',data:{labels:labels,datasets:[
  {label:'Bandwidth (GB/s)', data:bw, borderWidth:1, pointRadius:2.5, showLine:false,
   pointBackgroundColor:colors, pointBorderColor:colors}
]},options:{animation:false, scales:{x:{ticks:{maxTicksLimit:8}}, y:{suggestedMin:500,suggestedMax:700}}, plugins:{legend:{display:false}}}});

function autoscale(v){const y=chart.options.scales.y;if(v<y.suggestedMin+10||v>y.suggestedMax-10){y.suggestedMin=Math.floor(v-150);y.suggestedMax=Math.ceil(v+150);}}

ws.onmessage=(ev)=>{
  const msg=JSON.parse(ev.data);
  if(msg.type==='point'){
    labels.push(msg.chunk);
    bw.push(msg.gbps);
    modes.push(msg.mode);
    colors.push(palette[msg.mode]||'#1e88e5');
    if(labels.length>CFG.RING_SIZE){labels.shift();bw.shift();modes.shift();colors.shift();}
    autoscale(msg.gbps); chart.update('none');

    document.getElementById('bw').innerText=msg.gbps.toFixed(2);
    document.getElementById('errs').innerText=msg.total_errors;

    if (msg.per_mode_std){
      const s=msg.per_mode_std, get=(k)=> (k in s? Number(s[k]).toFixed(3):'--');
      document.getElementById('s1').innerText=get('1');
      document.getElementById('s2').innerText=get('2');
      document.getElementById('s3').innerText=get('3');
      document.getElementById('s4').innerText=get('4');
      document.getElementById('s5').innerText=get('5');
      document.getElementById('savg').innerText=(msg.avg_std||0).toFixed(3);
    }
  } else if(msg.type==='status'){
    document.getElementById('status').innerText=msg.text||msg.error||'';
  }
};

// 初始化控件默认值
document.getElementById('fraction').value = CFG.FRACTION.toFixed(2);
document.getElementById('iters').value = CFG.CHUNK_ITERS;
const slider=document.getElementById('memSlider'), label=document.getElementById('sliderLabel');
slider.min = CFG.SLIDER_MIN; slider.max = CFG.SLIDER_MAX; slider.step = CFG.SLIDER_STEP; slider.value = CFG.SLIDER_MIN;
function refreshLabel(){ label.innerText = '档位: ' + slider.value; } refreshLabel(); slider.oninput=refreshLabel;

// 控制按钮
document.getElementById('btnStart').onclick=async()=>{
  const f=parseFloat(document.getElementById('fraction').value||CFG.FRACTION);
  const it=parseInt(document.getElementById('iters').value||CFG.CHUNK_ITERS);
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fraction:f,iters:it})});
  const j=await r.json();document.getElementById('status').innerText=j.msg||'';
};
document.getElementById('btnStop').onclick=async()=>{
  const r=await fetch('/api/stop',{method:'POST'});
  const j=await r.json();document.getElementById('status').innerText=j.msg||'';
};
document.getElementById('btnRestart').onclick=async()=>{
  const f=parseFloat(document.getElementById('fraction').value||CFG.FRACTION);
  const it=parseInt(document.getElementById('iters').value||CFG.CHUNK_ITERS);
  const r=await fetch('/api/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fraction:f,iters:it})});
  const j=await r.json();document.getElementById('status').innerText=j.msg||'';
};

// 应用 Afterburner 档位（后端会清空统计）
document.getElementById('btnApply').onclick=async()=>{
  const payload={level:parseInt(slider.value)};
  const r=await fetch('/api/set_mem',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j=await r.json();document.getElementById('applyMsg').innerText=j.ok?('Applied: '+(j.cmd||'')):('Failed: '+(j.msg||'')); 
};
</script>
</body></html>
"""

@app.get("/")
async def index():
    cfg = {
        "FRACTION": FRACTION,
        "CHUNK_ITERS": CHUNK_ITERS,
        "RING_SIZE": RING_SIZE,
        "SLIDER_MIN": SLIDER_MIN,
        "SLIDER_MAX": SLIDER_MAX,
        "SLIDER_STEP": SLIDER_STEP,
    }
    html = INDEX_HTML.replace("__CFG_JSON__", json.dumps(cfg))
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
