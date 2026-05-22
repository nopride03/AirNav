import argparse
import json
import os
import threading
import urllib.error
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    image: str


def parse_backends():
    backend_urls = os.getenv("BACKEND_URLS")
    if backend_urls:
        return [url.strip().rstrip("/") for url in backend_urls.split(",") if url.strip()]

    gpu_list = os.getenv("GPU_LIST", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    base_port = int(os.getenv("BASE_PORT", "8050"))
    gpus = [gpu.strip() for gpu in gpu_list.split(",") if gpu.strip()]
    return [f"[REDACTED_URL] + i}" for i in range(len(gpus))]


app = FastAPI()
BACKENDS = parse_backends()
backend_index = 0
backend_lock = threading.Lock()


def next_backend():
    global backend_index
    with backend_lock:
        backend = BACKENDS[backend_index]
        backend_index = (backend_index + 1) % len(BACKENDS)
    return backend


def forward_generate(item: Item, timeout: float):
    errors = []
    attempted = set()
    payload = item.model_dump() if hasattr(item, "model_dump") else item.dict()

    while len(attempted) < len(BACKENDS):
        backend = next_backend()
        if backend in attempted:
            continue
        attempted.add(backend)

        request = urllib.request.Request(
            url=f"{backend}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            # Disable proxy for localhost/127.0.0.1 to avoid proxy interference
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{backend}: HTTP {exc.code} {detail}")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    raise HTTPException(status_code=503, detail={"message": "all backends failed", "errors": errors})


@app.get("/health")
def health():
    return {"status": "ok", "backends": BACKENDS}


@app.get("/backends")
def backends():
    return {"backends": BACKENDS}


@app.post("/generate")
async def generate(item: Item):
    return await run_in_threadpool(forward_generate, item, app.state.timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--backends", default="")
    args = parser.parse_args()

    if args.backends:
        BACKENDS = [url.strip().rstrip("/") for url in args.backends.split(",") if url.strip()]

    if not BACKENDS:
        raise ValueError("No backend instances configured. Set BACKEND_URLS or GPU_LIST.")

    app.state.timeout = args.timeout
    uvicorn.run(app, host=args.host, port=args.port)
