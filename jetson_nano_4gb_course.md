# Jetson Nano 4GB — 10-Session Mastery Plan

> Same style as the Docker plan: 80/20 focus, ~2 hours per session, concrete command blocks, checkpoints, honest tradeoffs. By the end you should be *fluent* on this board — you can flash it, tune it, debug it, containerize workloads on it, run TensorRT on it, and ship a portfolio-worthy project on it.
>
> Ends by porting your MiDaS server from the Pi 5 plan onto the Jetson with TensorRT acceleration — same repo, second target, benchmarks table. That comparison alone is a *killer* portfolio artifact.

---

## Reality check before we start (read this once)

The original Jetson Nano 4GB is legacy hardware. Being honest with you up front so nothing surprises you at Session 4:

| Thing | Reality |
|---|---|
| Latest supported JetPack | **4.6.6** (L4T 32.7.6). JP 5 / 6 / 7 do **not** support this board. |
| Ubuntu | 18.04. EOL for standard support. Python **3.6** default. |
| CUDA | **10.2**. cuDNN 8.2. TensorRT 8.2. |
| GPU | 128-core Maxwell. ~472 GFLOPS FP16. ~0.5 TFLOPS FP32. |
| RAM | 4 GB shared CPU + GPU. This is the real constraint. |
| Power | 5W or 10W mode. Thermal throttles hard without a fan. |
| Cameras | CSI (IMX219 Pi cam v2), USB. GStreamer is the native path. |
| Community | Mostly moved to Orin Nano. Some new tutorials skip this board. |

**Why bother, then?** Because (a) it's the canonical "I ran real CUDA inference on real edge silicon" portfolio piece, (b) every skill here — nvpmodel, nvidia-docker, TensorRT engine builds, jetson-inference, GStreamer camera pipelines — transfers directly to Orin, (c) it pairs perfectly with your MiDaS + navigation thesis narrative, and (d) the constraints *force* you to learn the optimization tricks (FP16, TensorRT, quantization) that hiring managers actually care about. On a beefier board you can be lazy. On a Nano you can't.

**What we deliberately skip** (the 20% not worth it for you): DLA cores (Nano doesn't have them), Yocto/BSP customization, custom TensorRT plugin authoring, hand-written CUDA kernels, GXF/Isaac ROS stuff that only runs on Orin. We stay on the load-bearing 80%.

---

## What you need on the desk

- Jetson Nano 4GB dev kit (B01 revision — the one with two CSI slots).
- **64 GB microSD** (SanDisk Extreme A2 or Samsung EVO Plus U3). Not a random 32 GB one.
- **5V/4A barrel-jack power supply** (not USB — micro-USB power throttles at 10W mode).
- **Jumper on J48** (next to barrel jack) so the board actually uses barrel power.
- Ethernet cable (WiFi needs a dongle; wire it for now).
- Host laptop with an SD card reader and Etcher/Rufus.
- Optional but recommended:
  - USB webcam (any UVC one) or a Raspberry Pi Camera v2 (IMX219).
  - USB 3.0 SSD (128 GB+) — we'll move Docker + swap onto it in Session 2. This is the single biggest quality-of-life upgrade.
  - A small fan on the heatsink. The stock passive sink throttles above ~65 °C.

---

## Session 1 — Flash, Boot, Live On It (2h)

**Goal:** Fresh JetPack 4.6.6 on the SD card, headless SSH access, `jtop` running, power mode set, essentials installed. You leave this session with a Jetson you can log into from your laptop and see the GPU on.

### Concepts (~20 min)
- What JetPack actually is: L4T (kernel + BSP) + CUDA + cuDNN + TensorRT + Multimedia API + samples. It's not just an OS.
- Why we use the **SD card image** and not SDK Manager for Nano (SDK Manager needs an Ubuntu 18.04/20.04 host; the SD image is a one-shot flash — simpler).
- `nv_tegra_release` vs JetPack version — L4T r32.7.6 = JetPack 4.6.6.
- Headless vs desktop-first boot. We go headless from day one to save ~800 MB RAM.

### Hands-on
```bash
# On your laptop:
# 1. Download SD card image from developer.nvidia.com/embedded/jetpack-archive
#    Pick: "Jetson Nano Developer Kit SD Card Image" for JetPack 4.6.6
# 2. Flash with balenaEtcher onto the 64 GB card. ~15 min.
# 3. Do NOT eject with an OS on it and boot yet — we'll do headless first boot.
```

**Headless first boot** (skip the desktop config screen, save RAM):
1. Insert SD, connect ethernet, keyboard optional but useful once.
2. Power on with the barrel jack (jumper J48 on).
3. On first boot the initial config wizard needs a display OR you can do the "headless serial" trick: connect micro-USB to your laptop, then `screen /dev/ttyACM0 115200` (Linux/Mac) or PuTTY (Windows) and complete the setup over serial.
4. Create your user (name it something short — `keerat` fine), set locale, keyboard, timezone.
5. When it asks about APP partition size — accept the default (uses all remaining space).

**Once you can SSH in from your laptop:**
```bash
# From your laptop
ssh keerat@<jetson-ip>

# First things, on the Jetson:
sudo apt update
sudo apt install -y nano curl wget git htop
cat /etc/nv_tegra_release          # Should say R32 (release), REVISION: 7.6
cat /etc/os-release                # Ubuntu 18.04.6 LTS

# Kill the desktop to reclaim RAM (~800 MB back)
sudo systemctl set-default multi-user.target
sudo reboot
```

**Install jtop — the single most useful tool on Jetson:**
```bash
sudo apt install -y python3-pip
sudo -H pip3 install -U jetson-stats
sudo reboot
jtop            # Live view of CPU, GPU, RAM, power, temp, all in one screen
```

**Set 10W power mode** (max perf, but ONLY if you're on barrel power):
```bash
sudo nvpmodel -m 0        # 0 = MAXN (10W), 1 = 5W
sudo jetson_clocks        # Locks clocks to max
nvpmodel -q               # Verify
```

### Checkpoint ✅
- [ ] `ssh` from laptop works without a monitor attached.
- [ ] `cat /etc/nv_tegra_release` shows `R32 (release), REVISION: 7.6`.
- [ ] `jtop` shows GPU frequency, memory, and temperature.
- [ ] `nvpmodel -q` shows `NV Power Mode: MAXN`.
- [ ] `free -h` shows ~3.7 GB total (headless).

### Tradeoffs
- Desktop is nice for the first day but kills 800 MB you'll want back at Session 9. Kill it now, use `jtop` and SSH.
- 5W mode is quieter/cooler but ~40% slower for inference. Default to 10W with a fan.

---

## Session 2 — Survival: Storage, Swap, Thermal, and Docker Data Root (2h)

**Goal:** Turn the Nano into a machine that won't die when you throw real workloads at it. This session isn't glamorous but everything else fails without it.

### Concepts (~15 min)
- 4 GB RAM is shared CPU + GPU. Loading a 1.5 GB PyTorch model already stresses it.
- **Swap is not optional** on Nano. Default swap is 2 GB zram — good but not enough. We add a file-backed swap on the SSD.
- SD card write endurance is finite. Move Docker's data root (which writes a lot) off the SD onto a USB SSD if you have one.
- Thermal: passive-cool Nano throttles at ~65 °C. A $10 5V fan solves it.

### Hands-on

**Check your baseline:**
```bash
free -h
swapon --show
df -h
sensors                       # If not installed: sudo apt install lm-sensors
tegrastats --interval 1000    # Ctrl-C to stop; watch CPU%, GPU%, RAM, temp
```

**Add 4 GB file-backed swap (on the SD card if no SSD; on SSD if you have one):**
```bash
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
swapon --show          # Should list both zram and /var/swapfile
```

**If you have a USB SSD, do this instead — huge upgrade:**
```bash
# Identify the SSD
lsblk
# Format ext4 (destroys data on the SSD!)
sudo mkfs.ext4 /dev/sda1
sudo mkdir /mnt/ssd
sudo mount /dev/sda1 /mnt/ssd

# Persistent mount
UUID=$(sudo blkid -s UUID -o value /dev/sda1)
echo "UUID=$UUID /mnt/ssd ext4 defaults 0 2" | sudo tee -a /etc/fstab

# Move swap to SSD
sudo swapoff /var/swapfile
sudo rm /var/swapfile
sudo fallocate -l 8G /mnt/ssd/swapfile
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile
sudo swapon /mnt/ssd/swapfile
sudo sed -i 's|/var/swapfile|/mnt/ssd/swapfile|' /etc/fstab
```

**Move Docker data root onto the SSD** (Docker is preinstalled on JetPack; images and containers otherwise fill the SD):
```bash
sudo systemctl stop docker
sudo mkdir -p /mnt/ssd/docker
# /etc/docker/daemon.json — create or edit
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "data-root": "/mnt/ssd/docker",
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
sudo systemctl start docker
docker info | grep -i "docker root dir"     # Confirm new path
```
(Note we also set `default-runtime: nvidia` — you'll thank yourself in Session 5.)

**Fan setup (if you have one):**
```bash
# 5V PWM fan on pins 4 & 6 works out of the box on B01 board
# Manual test:
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'   # Full speed
sudo sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'     # Off
# JetPack 4.6.x has jetson-fan-ctrl already for auto control
```

**Thermal stress test:**
```bash
# Terminal 1: watch stats
jtop
# Terminal 2: hammer the CPU + GPU briefly
sudo apt install -y stress-ng
stress-ng --cpu 4 --timeout 60s
# Watch for temperature and any throttle warnings in jtop
```

### Checkpoint ✅
- [ ] `swapon --show` shows at least 4 GB additional swap (8 GB on SSD is better).
- [ ] `df -h` shows plenty of free space and (if you did the SSD move) Docker root on `/mnt/ssd`.
- [ ] `docker info` shows `Default Runtime: nvidia` and `Docker Root Dir: /mnt/ssd/docker`.
- [ ] Under `stress-ng`, jtop stays under ~75 °C (with fan) or throttles gracefully (without).

### Tradeoffs
- File swap on SD works, but SD wear + slow. SSD is the real answer.
- Setting `default-runtime: nvidia` means *every* container gets GPU access whether you ask or not. Fine for a personal dev board. On multi-tenant hosts you'd flip this.

---

## Session 3 — The CUDA Stack: What's There, How to Prove It (2h)

**Goal:** Understand exactly what NVIDIA ships in JetPack, verify each layer works, run your first CUDA program, and get comfortable with the tools.

### Concepts (~25 min)
- **The stack:** L4T kernel → CUDA driver (built into kernel) → CUDA runtime 10.2 → cuDNN 8.2 → TensorRT 8.2 → VPI (vision) → Multimedia API → samples. All preinstalled at `/usr/local/cuda/`, `/usr/lib/aarch64-linux-gnu/`, `/opt/nvidia/`.
- **CUDA 10.2** is the ceiling here. This dictates every downstream choice: PyTorch 1.10 max, ONNX Runtime 1.11 wheels, TF 2.7 max.
- No `nvidia-smi` — that's for discrete GPUs. On Jetson use `tegrastats` and `jtop`.

### Hands-on

**Verify each layer:**
```bash
# CUDA compiler + version
nvcc --version                       # cuda_10.2

# cuDNN
dpkg -l | grep -i cudnn              # libcudnn8 8.2.x

# TensorRT
dpkg -l | grep -i tensorrt           # tensorrt 8.2.x
/usr/src/tensorrt/bin/trtexec --help | head

# Env — often not on PATH by default
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Run the classic first-CUDA-program on the actual device:**
```bash
# Copy samples to your home so make doesn't fail on read-only paths
cp -r /usr/local/cuda/samples ~/cuda-samples
cd ~/cuda-samples/1_Utilities/deviceQuery
make
./deviceQuery
# Expected: "Result = PASS", 128 CUDA cores, Compute Capability 5.3
```

**Run one more that touches memory bandwidth:**
```bash
cd ~/cuda-samples/1_Utilities/bandwidthTest
make
./bandwidthTest
# Expect ~20 GB/s device→device, ~2 GB/s host↔device (shared memory arch)
```

**trtexec sanity check — the TensorRT command-line tool you'll live in:**
```bash
# Build a tiny random engine just to prove trtexec works
/usr/src/tensorrt/bin/trtexec --help 2>&1 | head -40
# We'll use this for real in Session 7.
```

**Understand your device programmatically:**
```bash
python3 -c "
import subprocess
print(subprocess.check_output(['cat', '/proc/device-tree/model']).decode())
"
# Expected: 'NVIDIA Jetson Nano Developer Kit'
```

### Checkpoint ✅
- [ ] `nvcc --version` shows CUDA 10.2.
- [ ] `deviceQuery` outputs `Result = PASS`, 128 CUDA cores, CC 5.3.
- [ ] `trtexec --help` runs.
- [ ] You can articulate why `nvidia-smi` doesn't exist here.

### Tradeoffs
- CUDA 10.2 is old. You *cannot* pin newer PyTorch. Accept it and lean on containers (Session 5) to isolate.
- Compute Capability 5.3 (Maxwell) means many modern kernels (tensor cores, bf16) simply aren't available. Design around it — FP16 is your speed lever, not bf16.

---

## Session 4 — Python ML Stack on aarch64: The Wheel Dance (2h)

**Goal:** Get PyTorch, torchvision, ONNX Runtime, and OpenCV working — the right way. This is where 90% of beginners lose a weekend. You will not.

### Concepts (~20 min)
- Vanilla `pip install torch` gives you an **x86** wheel or a CPU-only aarch64 wheel with no CUDA. Both useless.
- NVIDIA ships **prebuilt aarch64 + CUDA 10.2 wheels** for PyTorch from a special URL (Qengineering and the NVIDIA forum thread by dusty-nv are the canonical sources).
- **torchvision** must be built from source matched to the torch version. `pip install torchvision` fetches a wheel that won't match.
- OpenCV: preinstalled version has no CUDA. For real work, either use it as-is or rebuild with CUDA (~2 hours; do only if you actually need CUDA-accelerated OpenCV, which for most CV work you don't).
- Python is 3.6. Some libraries won't install. Accept it or use containers (Session 5).

### Hands-on

**Virtual environment (isolation is sanity):**
```bash
sudo apt install -y python3-venv python3-dev libopenblas-dev libopenmpi-dev \
    libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev \
    libavformat-dev libswscale-dev
python3 -m venv ~/envs/ml
source ~/envs/ml/bin/activate
pip install --upgrade pip
```

**PyTorch 1.10 — NVIDIA's aarch64 CUDA wheel:**
```bash
# The canonical URL pattern (verify current file name on the NVIDIA forum thread
# "PyTorch for Jetson" if wget 404s — file names occasionally move)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
     -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Verify
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 1.10.0 True
```

**torchvision from source (~30 min compile):**
```bash
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python setup.py install --user
cd ..
python3 -c "import torchvision; print(torchvision.__version__)"
```

**ONNX + ONNX Runtime GPU (aarch64 wheel from Jetson Zoo):**
```bash
pip install onnx
# Get the ORT wheel matching JetPack 4.6 / CUDA 10.2
wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl \
     -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
pip install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Sanity: actually run a tensor on the GPU:**
```python
# quick_gpu.py
import torch, time
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
torch.cuda.synchronize()
t = time.time()
for _ in range(100):
    z = x @ y
torch.cuda.synchronize()
print(f"100 matmuls (1024²) on GPU: {(time.time()-t)*1000:.1f} ms")
```
```bash
python3 quick_gpu.py
# Expected: ~150–300 ms
```

### Checkpoint ✅
- [ ] `torch.cuda.is_available()` returns `True`.
- [ ] `torchvision` imports without a version mismatch warning.
- [ ] ONNX Runtime lists `TensorrtExecutionProvider` and `CUDAExecutionProvider`.
- [ ] `quick_gpu.py` finishes without OOM.

### Tradeoffs
- Building torchvision from source is annoying but the only path. Don't skip.
- You may hit `Killed` during builds — that's OOM. Add temporary swap and lower `-j` (single-threaded build: `python setup.py install`).
- **If the whl URLs 404:** the source of truth is the pinned "PyTorch for Jetson" thread on forums.developer.nvidia.com — search there for the current URL. Skip this entire session by using L4T Docker containers (next session).

---

## Session 5 — nvidia-docker on Jetson: Where Docker Skills Pay Off (2h)

**Goal:** Everything you already know about Docker, plus the one Jetson-specific piece (`nvidia` runtime + L4T base images). Once you get this, you never fight the aarch64 wheel dance again.

### Concepts (~15 min)
- **NVIDIA Container Toolkit** is preinstalled on JetPack. That's the `nvidia-container-runtime` you configured in Session 2.
- **L4T base images** ship on NGC (nvcr.io). They already have CUDA/cuDNN/TensorRT for your L4T version. You do *not* rebuild the stack — you `FROM nvcr.io/nvidia/l4t-...` and go.
- Key images for JetPack 4.6.x:
  - `nvcr.io/nvidia/l4t-base:r32.7.1` — bare minimum, CUDA + driver mount.
  - `nvcr.io/nvidia/l4t-ml:r32.7.1-py3` — PyTorch, TF, JupyterLab, scikit-learn preinstalled.
  - `nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3` — leaner, just PyTorch.
  - `nvcr.io/nvidia/l4t-tensorrt:r32.7.1-runtime` — TensorRT + Python bindings.
- The tag must match your L4T release. `r32.7.1` works for L4T 32.7.x boards.

### Hands-on

**Confirm GPU passes through to a container:**
```bash
# --runtime=nvidia not needed because you set default-runtime in Session 2
docker run --rm -it nvcr.io/nvidia/l4t-base:r32.7.1 \
    bash -c "nvcc --version && ls /usr/local/cuda/lib64"
# Should see CUDA 10.2 libraries mounted from the host
```

**Run a full PyTorch container — no wheel install needed:**
```bash
docker run --rm -it \
    --network host \
    -v $HOME:/workspace \
    nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 \
    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Run the ML container with JupyterLab exposed to your laptop:**
```bash
docker run --rm -it \
    --network host \
    -v $HOME:/workspace \
    nvcr.io/nvidia/l4t-ml:r32.7.1-py3
# Inside container:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# From your laptop browser: http://<jetson-ip>:8888  (token from stdout)
```

**Build your own Dockerfile on top of L4T PyTorch** — this is the pattern for MiDaS:
```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir fastapi uvicorn[standard] pillow numpy

COPY app/ /app/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
docker build -t jetson-midas:dev .
docker run --rm -p 8000:8000 jetson-midas:dev
```

**Camera passthrough — you'll need this in Session 6:**
```bash
docker run --rm -it \
    --device /dev/video0 \
    -v /tmp/argus_socket:/tmp/argus_socket \
    nvcr.io/nvidia/l4t-base:r32.7.1 bash
```

### Checkpoint ✅
- [ ] `nvcc --version` runs inside an L4T container.
- [ ] `torch.cuda.is_available()` is `True` inside `l4t-pytorch`.
- [ ] JupyterLab in `l4t-ml` reachable from your laptop.
- [ ] You built one custom image on top of an L4T base and it ran.

### Tradeoffs
- L4T base images are **big** (2–5 GB). Cross-arch builds from x86 need `--platform linux/arm64` and QEMU (we cover this in Session 10 CI).
- The runtime *mounts* CUDA from the host — it's not inside the image. That means the container's CUDA version follows your L4T version. Portable across the same L4T family, not across major versions.
- If you ever move to Orin: the workflow is identical, just swap `r32.7.1` for the new L4T tag.

---

## Session 6 — Cameras & GStreamer, the Jetson Way (2h)

**Goal:** Get a live camera into Python/OpenCV, with the GStreamer pipeline you'll use for every real project. USB or CSI, both covered.

### Concepts (~20 min)
- On Jetson, the "right" camera path is **GStreamer** with hardware-accelerated elements (`nvarguscamerasrc` for CSI, `v4l2src` for USB, `nvvidconv` for zero-copy format conversion). Not `cv2.VideoCapture(0)` with a plain index — that works for USB but leaves perf on the table.
- CSI camera detection: `ls /dev/video*` (should show `/dev/video0` after boot with camera attached).
- Argus daemon runs in the background and mediates CSI cameras — that's what `/tmp/argus_socket` is.

### Hands-on

**USB webcam first (simplest):**
```bash
ls /dev/video*                  # /dev/video0 should exist
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
# Try a raw capture
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! xvimagesink
# Ctrl-C to stop. If headless, replace xvimagesink with fakesink and add ! identity silent=false
```

**CSI camera (IMX219, Pi cam v2 clone or genuine):**
```bash
# Bring up the argus test app
nvgstcapture-1.0
# Or a manual pipeline:
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
    nvvidconv ! 'video/x-raw,format=BGRx' ! videoconvert ! \
    'video/x-raw,format=BGR' ! xvimagesink
```

**In Python + OpenCV (the pattern you'll reuse everywhere):**
```python
# camera.py
import cv2

def gst_csi_pipeline(w=1280, h=720, fps=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )

def gst_usb_pipeline(dev="/dev/video0", w=640, h=480, fps=30):
    return (
        f"v4l2src device={dev} ! "
        f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )

cap = cv2.VideoCapture(gst_usb_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

n = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    n += 1
    if n % 30 == 0: print(f"Read {n} frames, shape={frame.shape}")
    if n > 300: break

cap.release()
```
Run it:
```bash
python3 camera.py
```

**Save a frame to prove it works headless:**
```python
# Add this before releasing
cv2.imwrite('/tmp/frame.jpg', frame)
```
```bash
scp keerat@<jetson-ip>:/tmp/frame.jpg .   # From your laptop
```

### Checkpoint ✅
- [ ] `gst-launch-1.0` shows video (or captures without error headless).
- [ ] `camera.py` reads at least 300 frames without stalling.
- [ ] You can articulate why `cv2.VideoCapture(0)` differs from a `nvarguscamerasrc` pipeline.

### Tradeoffs
- `nvarguscamerasrc` only works for CSI cameras with an Argus driver (IMX219, IMX477 with the driver). Generic USB webcams go through `v4l2src`.
- GStreamer pipelines are picky about caps (format strings). When they fail, `GST_DEBUG=3 gst-launch-1.0 ...` shows the mismatch.

---

## Session 7 — TensorRT: The Reason We're On Jetson (2h)

**Goal:** Take a PyTorch model → ONNX → TensorRT engine → run it and benchmark. This is the skill that separates "I ran a model on Jetson" from "I optimized a model for Jetson."

### Concepts (~25 min)
- **TensorRT** is NVIDIA's inference compiler. It takes a network, fuses layers, picks kernels, and emits a device-specific engine (`.engine` / `.plan`).
- **FP16** on Nano's Maxwell GPU gives ~1.5–2x speedup over FP32, with negligible accuracy loss for most CV models.
- **INT8** needs calibration data and is finicky on Maxwell. FP16 is your default lever.
- **Two ways to build an engine:**
  1. `trtexec` command-line — the fast path, great for benchmarking.
  2. Python API (`tensorrt` module) — for programmatic build in your app.
- **Engines are hardware-specific.** An engine built on Nano won't run on Orin. Rebuild per device.

### Hands-on

**Export a small model to ONNX:**
```python
# export_onnx.py
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True).eval().cuda()
dummy = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(
    model, dummy, 'resnet18.onnx',
    input_names=['input'], output_names=['output'],
    opset_version=11,
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
)
print("Exported resnet18.onnx")
```

**Build a TensorRT engine with `trtexec` — FP32 baseline first:**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=resnet18.onnx \
    --saveEngine=resnet18_fp32.engine \
    --workspace=1024 \
    --verbose 2>&1 | tail -30
```
Look at the "GPU Compute Time" summary at the end — that's your latency.

**Now FP16 (the whole point):**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=resnet18.onnx \
    --saveEngine=resnet18_fp16.engine \
    --workspace=1024 \
    --fp16
# Compare GPU Compute Time to FP32 — expect ~1.5–2x speedup
```

**Load and run the engine in Python:**
```python
# run_engine.py
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open('resnet18_fp16.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

ctx = engine.create_execution_context()
inp_shape = engine.get_binding_shape(0)
if inp_shape[0] == -1: inp_shape = (1,) + tuple(inp_shape[1:])
out_shape = engine.get_binding_shape(1)
if out_shape[0] == -1: out_shape = (1,) + tuple(out_shape[1:])

inp = np.random.rand(*inp_shape).astype(np.float32)
out = np.empty(out_shape, dtype=np.float32)

d_in = cuda.mem_alloc(inp.nbytes)
d_out = cuda.mem_alloc(out.nbytes)
stream = cuda.Stream()

cuda.memcpy_htod_async(d_in, inp, stream)
ctx.execute_async_v2([int(d_in), int(d_out)], stream.handle)
cuda.memcpy_dtoh_async(out, d_out, stream)
stream.synchronize()
print("Output shape:", out.shape, "top-5 idx:", out[0].argsort()[-5:][::-1])
```
```bash
pip install pycuda           # If not already
python3 run_engine.py
```

**Benchmark against raw PyTorch — this is the number that goes in your README:**
```bash
# Time PyTorch FP32
python3 -c "
import torch, time, torchvision
m = torchvision.models.resnet18(pretrained=True).eval().cuda().half()
x = torch.randn(1,3,224,224,device='cuda').half()
for _ in range(10): m(x)
torch.cuda.synchronize(); t=time.time()
for _ in range(100): m(x)
torch.cuda.synchronize()
print(f'PyTorch FP16: {(time.time()-t)*10:.2f} ms/inference')
"

# Time TensorRT via trtexec
/usr/src/tensorrt/bin/trtexec --loadEngine=resnet18_fp16.engine --iterations=200
# Report the "mean" GPU Compute Time
```

### Checkpoint ✅
- [ ] ONNX export succeeds, ONNX Runtime can load it.
- [ ] `trtexec` builds both an FP32 and FP16 engine without errors.
- [ ] Python script loads the engine and produces sensible outputs.
- [ ] You have a real number: "TensorRT FP16 is Xx faster than PyTorch on my Nano." Write it down.

### Tradeoffs
- Engine building is slow on Nano (5–20 min for a modest model). Do it once per hardware; ship the engine.
- Some layers/ops aren't supported by TRT 8.2 — you'll see an error naming the op. Two options: (a) rewrite the model to avoid it, (b) fall back to CUDA execution for that op via ONNX Runtime with TRT provider (Session 9).

---

## Session 8 — jetson-inference: Standing on dusty-nv's Shoulders (2h)

**Goal:** Understand and use the community library that wraps TensorRT for the common CV tasks — classification, detection, segmentation, pose, depth. It's a reference implementation you can also read for how a well-structured TensorRT app looks.

### Concepts (~15 min)
- `jetson-inference` (by NVIDIA's Dustin Franklin) is a C++ library with Python bindings that wraps TensorRT for canonical CV workloads.
- It handles: engine caching, camera input, overlay rendering, model downloads.
- You will use it directly for prototyping, and read it for pattern (their `imageNet.py`, `detectNet.py`, `segNet.py`, and `depthNet.py` are excellent references).

### Hands-on

**Build from source (they don't ship apt packages for L4T 32.7):**
```bash
sudo apt install -y libpython3-dev python3-numpy cmake
git clone --recursive --branch L4T-R32.7.1 \
    https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build && cd build
cmake ../
# During cmake it will offer a model download menu — pick:
#   ResNet-18, SSD-Mobilenet-v2, FCN-ResNet18-Cityscapes-512x256, MonoDepth-fcn-mobilenet
# It will also ask about installing PyTorch — skip (we already have it)
make -j$(nproc)
sudo make install
sudo ldconfig
```

**Run classification on an image:**
```bash
cd ~/jetson-inference/build/aarch64/bin
./imagenet.py --network=resnet-18 images/orange_0.jpg /tmp/out.jpg
```

**Run object detection on the camera (the demo hiring managers actually understand):**
```bash
./detectnet.py --network=ssd-mobilenet-v2 /dev/video0
# Or headless with recorded output:
./detectnet.py --network=ssd-mobilenet-v2 /dev/video0 rtsp://@:8554/stream
# Then vlc rtsp://<jetson-ip>:8554/stream on your laptop
```

**Monocular depth — the direct thesis-relevant one:**
```bash
./depthnet.py --network=mono-depth --visualize=input,depth \
    images/room_1.jpg /tmp/depth_out.jpg
```

**Read the source for the pattern:** Open `~/jetson-inference/python/examples/depthnet.py` — that's ~50 lines of "load engine, grab frame, infer, visualize" that you'll structure your MiDaS server around.

### Checkpoint ✅
- [ ] `imagenet.py` classifies the sample orange correctly.
- [ ] `detectnet.py` shows bounding boxes on your webcam feed (RTSP or preview).
- [ ] `depthnet.py` produces a plausible depth map.
- [ ] You've read one of their example scripts end to end.

### Tradeoffs
- Building takes ~30 min on Nano. Worth it once.
- Their model zoo is dated; MiDaS is not in it. That's why we bring our own MiDaS in the next session.

---

## Session 9 — Portfolio Project Part 1: MiDaS on Jetson with TensorRT (2h)

**Goal:** Port your MiDaS FastAPI server from the Pi 5 plan onto the Jetson, replace ONNX Runtime CPU inference with **TensorRT FP16** inference, benchmark, containerize. This session is where the two portfolio projects fuse into one story.

### The plan
- Same repo, add a `jetson/` subdirectory (or a `target` config).
- Same FastAPI endpoint (`POST /depth` with an image, returns depth PNG + JSON).
- Reuse the MiDaS ONNX model from the Pi 5 project.
- Build a TensorRT engine from that ONNX at container start (cached), then serve.

### Hands-on

**Get MiDaS-small (the version that fits in 4 GB and runs fast):**
```bash
mkdir -p ~/jetson-midas && cd ~/jetson-midas
# You already have this from the Pi 5 project — copy it over:
# scp midas_small.onnx keerat@<jetson-ip>:~/jetson-midas/
# Or download fresh:
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/midas_v21_small_256.onnx \
    -O midas_small.onnx
```

**Build the engine once (cache it for the container):**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=midas_small.onnx \
    --saveEngine=midas_small_fp16.engine \
    --fp16 \
    --workspace=1024 \
    --verbose 2>&1 | tail -20
# Note the mean GPU Compute Time — that's your inference latency.
```

**FastAPI server using the engine:**
```python
# app/main.py
import io, time
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "/models/midas_small_fp16.engine"
INPUT_SIZE = 256

class TRTDepth:
    def __init__(self, path):
        with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.d_in = cuda.mem_alloc(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
        self.d_out = cuda.mem_alloc(1 * INPUT_SIZE * INPUT_SIZE * 4)

    def infer(self, img: np.ndarray) -> np.ndarray:
        # img: HxWx3 uint8 RGB
        x = img.astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = x.transpose(2, 0, 1)[None].astype(np.float32).copy()
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.execute_async_v2([int(self.d_in), int(self.d_out)], self.stream.handle)
        out = np.empty((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out

app = FastAPI()
depth = TRTDepth(ENGINE_PATH)

@app.post("/depth")
async def depth_endpoint(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    t = time.time()
    d = depth.infer(np.array(img))
    latency_ms = (time.time() - t) * 1000
    return JSONResponse({
        "shape": list(d.shape),
        "min": float(d.min()),
        "max": float(d.max()),
        "latency_ms": round(latency_ms, 2),
    })

@app.get("/healthz")
def healthz(): return {"ok": True}
```

**Dockerfile for the Jetson variant:**
```dockerfile
# jetson/Dockerfile
FROM nvcr.io/nvidia/l4t-tensorrt:r32.7.1-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-libnvinfer python3-pycuda \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir fastapi "uvicorn[standard]" pillow numpy

COPY app/ /app/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml — mount the prebuilt engine:**
```yaml
services:
  depth:
    build:
      context: .
      dockerfile: jetson/Dockerfile
    image: jetson-midas:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    restart: unless-stopped
```

**Run it and hit it:**
```bash
cp midas_small_fp16.engine models/
docker compose up --build -d
docker compose logs -f depth        # Watch startup
# From your laptop:
curl -X POST -F "file=@test_image.jpg" http://<jetson-ip>:8000/depth
```

### Checkpoint ✅
- [ ] `curl` to `/depth` returns JSON with a `latency_ms` field.
- [ ] Latency is <100 ms per image (should be closer to 30–50 ms FP16).
- [ ] Container survives a restart (`docker compose restart`).
- [ ] You have a concrete comparison number: Pi 5 ONNX CPU vs Jetson Nano TensorRT FP16.

### Tradeoffs
- We're loading a fixed-size engine. Add dynamic shapes if you want to accept variable resolutions (build with `--minShapes/--optShapes/--maxShapes` in `trtexec`).
- Engine built inside the container will be device-locked. Building on the host and mounting (as we did) is cleaner and lets you rebuild without a container rebuild.

---

## Session 10 — Portfolio Project Part 2: Ship It (2h)

**Goal:** Turn the working project into something a hiring manager can look at in 3 minutes and think "yes, this person deploys to edge." Live camera, benchmarks, systemd, GitHub, README.

### Hands-on

**Live camera → depth pipeline (headless-friendly):**
```python
# app/live.py — a separate process that grabs frames and posts to the depth endpoint
import cv2, requests, time
from camera import gst_usb_pipeline   # From Session 6

cap = cv2.VideoCapture(gst_usb_pipeline(w=256, h=256, fps=15), cv2.CAP_GSTREAMER)
while True:
    ret, frame = cap.read()
    if not ret: break
    _, buf = cv2.imencode('.jpg', frame)
    t = time.time()
    r = requests.post('http://localhost:8000/depth',
                      files={'file': ('f.jpg', buf.tobytes(), 'image/jpeg')})
    print(f"{r.json()['latency_ms']:.1f} ms server, {(time.time()-t)*1000:.1f} ms round-trip")
```

**Systemd service so it starts on boot:**
```bash
sudo tee /etc/systemd/system/jetson-midas.service > /dev/null <<'EOF'
[Unit]
Description=Jetson MiDaS depth server
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/home/keerat/jetson-midas
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-midas
sudo systemctl status jetson-midas
```

**Benchmarks table — this is the portfolio gold:**
```bash
# Bench script — run each once, capture median of 100 iterations
python3 bench.py --backend=torch-fp32
python3 bench.py --backend=torch-fp16
python3 bench.py --backend=onnxruntime-cuda
python3 bench.py --backend=onnxruntime-tensorrt
python3 bench.py --backend=tensorrt-fp16
```
Assemble a table for the README:

| Backend | Device | Latency (ms) | Throughput (fps) | Notes |
|---|---|---|---|---|
| ONNX CPU | Pi 5 | ~X | ~Y | From Pi 5 project |
| PyTorch FP32 | Nano | ~X | ~Y | Baseline |
| PyTorch FP16 | Nano | ~X | ~Y | half() |
| ONNX Runtime CUDA | Nano | ~X | ~Y | |
| ONNX Runtime TensorRT | Nano | ~X | ~Y | EP="TensorrtExecutionProvider" |
| **TensorRT FP16** | Nano | ~X | ~Y | **Ship this** |

**README structure that lands:**
1. **One-line pitch** — "Monocular depth-estimation server for the Jetson Nano 4GB, 3.4× faster than Pi 5 baseline via TensorRT FP16."
2. **Screenshot/GIF** — live camera → depth overlay. Record with `nvidia-smi`-style latency in a corner.
3. **Architecture diagram** — client → FastAPI → TensorRT engine → GPU. 5 boxes, arrows, done.
4. **Quickstart** — one `docker compose up` after flashing.
5. **Benchmarks table** — the one above.
6. **How the engine is built** — the `trtexec` command with FP16 flag.
7. **Deployment** — systemd unit shown.
8. **What I learned** — 3 bullets, honest.

**GitHub Actions on aarch64 — the honest tradeoff:**
- GitHub-hosted runners are x86. Cross-building an L4T image with QEMU works but is *slow* (~30 min).
- Two realistic paths:
  1. **Only lint / test / build in CI**, ship the actual Docker image build to be done on-device with a documented `make deploy`.
  2. **Self-hosted runner on the Jetson itself** — GitHub Actions has a self-hosted runner agent for aarch64. Install it, tag it `jetson`, and jobs `runs-on: [self-hosted, jetson]`.

Add a minimal `.github/workflows/ci.yml` that lints Python and validates the Dockerfile syntax (x86), and note in the README that image builds happen on-device (or via self-hosted runner). Being honest about this in the README is a *strength*, not a weakness.

### Checkpoint ✅
- [ ] Live camera → depth pipeline runs at 10+ fps.
- [ ] Systemd service starts on boot; `curl` works after a reboot with no manual steps.
- [ ] Benchmarks table filled in with real numbers.
- [ ] Repo has a README with pitch, GIF/screenshot, table, and quickstart.
- [ ] You have pushed to GitHub with the Jetson variant on a branch or as `jetson/` subdir alongside the Pi 5 work.

### Tradeoffs
- Real-time (30 fps) live depth needs shortcuts: shared-memory instead of HTTP, or in-process instead of two containers. Fine — document that as "future work" and ship 10 fps.
- The whole GH Actions aarch64 story is genuinely awkward; mentioning it explicitly reads as competent, not as a gap.

---

## After the course — where to go next

Now that you're fluent on Nano, three natural extensions that all reuse this base:

1. **ROS 2 bridge on Jetson** — subscribe to your quadruped's camera topic, publish a depth topic. This is your thesis fusing with your portfolio.
2. **INT8 calibration** — build a small calibration dataset, quantize the MiDaS engine to INT8, benchmark. Almost nobody does this well; it's a differentiator.
3. **Port to Orin Nano** — buy time on a rented Orin or an actual board. Nearly all your code moves as-is; only the L4T tag changes. Same benchmarks table, third column — that's a *very* strong portfolio artifact.

## Anti-patterns to avoid

- Powering from micro-USB while claiming to run at 10W — it silently caps power and every benchmark you post is invalid.
- Reporting inference time that includes preprocessing (JPEG decode, resize, normalize). Report GPU compute time separately; total pipeline time separately.
- Skipping the swap file until it's too late. When you OOM mid-build there's no recovery — you re-flash.
- Building torch/torchvision from source at every session. Use containers.
- Any variant of "Jetson Nano is dead, why did I do this" in the README. Frame it as: "targeted the constrained platform to force the optimization skills."

## Quick reference card

```bash
# Every session starts with:
jtop                             # Live monitor (Ctrl-C to exit)
nvpmodel -q                      # Confirm 10W mode
sudo jetson_clocks                # Max clocks
tegrastats --interval 500         # Text-mode monitor

# Docker with GPU (default-runtime nvidia set in Session 2)
docker run --rm nvcr.io/nvidia/l4t-base:r32.7.1 nvcc --version

# TensorRT engine build
/usr/src/tensorrt/bin/trtexec --onnx=m.onnx --saveEngine=m.engine --fp16

# Verify CUDA in Python
python3 -c "import torch; print(torch.cuda.is_available())"

# L4T / JetPack version
cat /etc/nv_tegra_release
```

---

*Same rules as the Docker plan: don't skip checkpoints, don't skim command blocks, keep a scratch `.md` per session with what surprised you. When Session 9 lands, take a picture of the board with the camera plugged in and a depth map on your laptop — that photo is going in your CV.*
