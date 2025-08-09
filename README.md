GDDR7 Tester UI — Plain Text README

Overview
--------
This tool displays GDDR6/GDDR7 memory bandwidth and jitter in your local
browser to help find a stable overclock quickly.  The page exposes Start /
Stop / Restart controls and can adjust memory clocks directly through
NVAPI/NVML or by applying MSI Afterburner profiles.  Statistics reset
whenever the frequency changes so the new setting stabilizes faster.

Features
--------
* Live scatter plot cycling through five access modes (Mode 1–5) with
  color‑coded points.
* Rolling standard deviation (σ) for each mode plus an overall average σ̄.
* Web controls to start, stop, and restart the tester.
* Memory clock changes via NVAPI/NVML with optional MSI Afterburner
  integration (Profiles 1–5).  Stats clear automatically after applying a
  new setting.
* The CUDA tester uses 128‑bit read/write with A↔B pattern flips; every
  access mode is a permutation so no address conflicts occur.

Environment
-----------
* Windows 11 with the latest NVIDIA drivers (validated on RTX 5070 / 5090).
* Python 3.10+ for the local web UI.
* Optional: `pynvml` for NVAPI/NVML control or MSI Afterburner configured
  with Profiles 1–5.

Quick Start
-----------
```
cd C:\gddr7_tester
py -m pip install "uvicorn[standard]" fastapi websockets
py ui_server.py
```
Open <http://127.0.0.1:8000> in your browser.

On the page:
* Use Start / Stop / Restart to control the tester.
* Adjust the memory clock offset with the slider or pick an Afterburner
  profile and click **Apply**; stats reset on success.
* Monitor per‑mode σ and overall σ̄—sharp increases or new errors indicate
  instability.

Test Details
------------
* Mode 1: linear (sequential)
* Mode 2: stride 64 KiB
* Mode 3: stride 128 KiB
* Mode 4: block‑xor (perturb within 4 KiB blocks; tail blocks use identity
  mapping)
* Mode 5: permute (multiplicative permutation)

Each kernel iteration performs a read + write and flips between two patterns
for verification.  The UI keeps a rolling window per mode to compute σ and
shows the average σ̄.  When a new frequency is applied (NVAPI or Afterburner)
the stats are cleared so the new setting converges quickly.

Building from Source (optional)
-------------------------------
Standard build (target machine requires CUDA runtime DLL):
```
nvcc -O3 --std=c++17 gddr7_tester.cu -o gddr7_tester.exe -Xptxas -dlcm=cg --compiler-options="/O2 /MD /EHsc"
```
Single‑file distribution (no CUDA Toolkit needed on target machine):
```
nvcc -O3 --std=c++17 gddr7_tester.cu -o gddr7_tester.exe -Xptxas -dlcm=cg -cudart=static \
  --compiler-options="/O2 /MT /EHsc" -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
```
Notes:
* `-cudart=static` links the CUDA runtime statically.
* `/MT` uses the static CRT to avoid VC runtime dependencies.
* Includes both `sm_120` (Ada) and `compute_120` (PTX); the first run on new
drivers may have a one‑time JIT delay.

Suggested Project Layout
------------------------
* `ui_server.py` – local web UI (FastAPI + WebSocket + Chart.js)
* `gddr7_tester.cu` – CUDA tester source
* `gddr7_tester.exe` – precompiled Windows binary (optional)
* `requirements.txt` – Python dependencies
* `assets\` – screenshots or demo images (optional)

License
-------
MIT

Contributing
------------
Issues and pull requests are welcome.
