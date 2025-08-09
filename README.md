GDDR7 Tester UI — Plain Text README

简介
本工具在本机浏览器里实时展示显存（GDDR6/GDDR7）带宽与抖动，帮助你在超频时快速找到“甜蜜点”。页面提供 Start / Stop / Restart 控制，并可直接通过 NVAPI/NVML 调整显存频率，或配合 MSI Afterburner 一键切换显存档位。切换频率后统计会自动清零，让新频率更快稳定。

主要功能

实时散点图，5 种访问模式（Mode 1→5）循环采样，不同模式用不同颜色区分。

统计每个模式的滚动标准差（σ），并计算平均 σ̄（各模式 σ 的平均）。

网页端按钮：开始、停止、重启测试器。

支持 NVAPI/NVML 调用直接设置显存频率；也提供 MSI Afterburner 集成，可一键应用 Profile 1–5 并在应用后清空统计窗口。

测试程序使用 128-bit 读写与 A↔B 图案翻转校验；所有访问模式都是全排列（无地址冲突），避免误报。

运行环境

Windows 11 和最新版 NVIDIA 显卡驱动（已在 RTX 5070 / 5090 上验证）

Python 3.10 或更高（仅用于本地网页 UI）

可选：安装 `pynvml`（使用 NVAPI/NVML 直接调节显存）或 MSI Afterburner（需预先配置 Profile 1–5）

快速开始（推荐）

在命令行进入项目目录，例如：
cd C:\gddr7_tester

安装 UI 依赖：
py -m pip install "uvicorn[standard]" fastapi websockets

启动本地网页 UI：
py ui_server.py

浏览器打开：
http://127.0.0.1:8000

在页面上：

使用 Start / Stop / Restart 控制测试器

通过滑块设置显存频率偏移或选择 Afterburner 档位并点击“应用”（成功后统计清零）

观察同一模式的 σ 和总体 σ̄；若 σ 明显增大或出现 New errors，则频率不稳

测试细节

Mode 1: linear（顺序）

Mode 2: stride 64 KiB

Mode 3: stride 128 KiB

Mode 4: block-xor（4 KiB 块内扰动；尾块使用恒等映射）

Mode 5: permute（乘法置换）

每次 kernel 对每个元素执行读 + 写，并在两种图案之间翻转校验。

UI 为每个模式维护独立的滚动窗口，计算各自 σ；同步显示平均 σ̄。

应用新的频率（NVAPI 或 Afterburner）后，后台会清空统计窗口，以便新的频率快速收敛。

从源码构建（可选）
普通构建（目标机需要 CUDA 运行时 DLL）：
nvcc -O3 --std=c++17 gddr7_tester.cu -o gddr7_tester.exe -Xptxas -dlcm=cg --compiler-options="/O2 /MD /EHsc"

单文件分发（推荐，目标机无需安装 CUDA Toolkit）：
nvcc -O3 --std=c++17 gddr7_tester.cu -o gddr7_tester.exe -Xptxas -dlcm=cg -cudart=static --compiler-options="/O2 /MT /EHsc" -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
说明：

-cudart=static 把 CUDA runtime 静态链接进 EXE

/MT 使用静态 CRT，避免 VC 运行库缺失

同时包含 sm_120（Ada）与 compute_120（PTX），首次在新驱动上可能有一次 JIT 延迟

项目结构建议

ui_server.py 本地网页 UI（FastAPI + WebSocket + Chart.js）

gddr7_tester.cu CUDA 测试程序源码

gddr7_tester.exe 已编译的 Windows 可执行文件（可选放仓库）

requirements.txt Python 依赖（见下）

assets\ 放置截图或演示图（可选）

许可证
MIT

联系方式
欢迎提 Issue 或 PR。
