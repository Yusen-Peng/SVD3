


```bash
export TORCH_LIB_DIR=/data/yusenp/.conda/envs/tsb-fcst/lib/python3.10/site-packages/torch/lib
unset LD_PRELOAD
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
NVJIT_PATH="$(python - <<'PY'
import sys, os, glob
for p in sys.path:
    if 'site-packages' in p:
        c = glob.glob(os.path.join(p,'nvidia','nvjitlink','lib','libnvJitLink.so.12*'))
        if c:
            print(c[0]); break
PY
)"
unset LD_PRELOAD
export LD_PRELOAD="$NVJIT_PATH"
export LD_LIBRARY_PATH="$(dirname "$NVJIT_PATH"):${LD_LIBRARY_PATH:-}"
python -c "import torch; print(torch.cuda.is_available())"
```