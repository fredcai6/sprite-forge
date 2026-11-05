# Build context: a ComfyUI checkout at comfy/COMFY_REF. Source and weights are mounted at run time.
FROM rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio)$' /tmp/requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt
