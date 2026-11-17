"""Compare streaming GRUCell with batched input projections and contributions.

Run from the repository root: python -m examples.ibeamnet.benchmark_forward
Only small synthetic inputs and the cache's JSON description are read.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import time
import torch
from .model import RecurrentSurrogate

ROOT = Path(__file__).resolve().parents[2]


def reference_encode(network, inputs):
    """Original per-layer embedding, GRUCell and contribution execution."""
    hidden = inputs.new_zeros((inputs.shape[0], network.hidden_size))
    total = inputs.new_zeros((inputs.shape[0], network.contribution_size))
    for layer in inputs.unbind(1):
        active = layer.ne(0).any(dim=1, keepdim=True).to(inputs.dtype)
        next_hidden = network.gru(network.ad_embedding(layer), hidden)
        total = total + network.contribution(torch.cat((layer, next_hidden), dim=1)) * active
        hidden = hidden * (1 - active) + next_hidden * active
    return torch.cat((total, hidden), dim=1)


def reference_forward(network, inputs):
    latent = reference_encode(network, inputs)
    return torch.cat([head(latent) for head in network.heads], dim=1)


def measure(function, device, target_seconds=.1):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(3):
        function()
    sync()
    start = time.perf_counter()
    function()
    sync()
    single = time.perf_counter() - start
    iterations = max(1, min(100, int(target_seconds / max(single, 1e-6))))
    samples = []
    for _ in range(3):
        sync()
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        sync()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.median(samples)


def notebook_options():
    notebook = json.loads((ROOT/"examples/ibeamnet/train/train.ipynb").read_bytes())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "MODEL_OPTIONS =" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_OPTIONS" for t in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError("Notebook has no literal MODEL_OPTIONS mapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT/"examples/ibeamnet/artifacts/forward-speed.json")
    args = parser.parse_args()
    if args.layers < 1 or min(args.batches) < 1 or args.cpu_threads < 1:
        parser.error("Layer/batch/thread counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((ROOT/"examples/ibeamnet/cache/manifest.json").read_bytes())
    settings = notebook_options()
    network = RecurrentSurrogate(manifest["elements"], [s["channels"] for s in manifest["spectra"]], **settings).eval()
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    report = dict(torch=torch.__version__, layers=args.layers, elements=len(manifest["elements"]),
                  channels=network.spectra_channels, model_options=settings, cpu_threads=args.cpu_threads,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  dtype="float32", tf32=False, recurrence_kind="gru", measurements=[])
    print(json.dumps({k:v for k,v in report.items() if k != "measurements"}, indent=2), flush=True)
    with torch.inference_mode():
        for device in devices:
            network.to(device)
            for batch in args.batches:
                inputs = torch.rand(batch, args.layers, len(network.elements), device=device)
                if args.layers > 2:
                    inputs[:, 1] = 0  # Interior empty layer.
                    inputs[0, -2:] = 0  # Trailing padding in one row.
                for scope, baseline, optimized in (("encoder", reference_encode, lambda m,x:m.encode_layers(x)),
                                                    ("full", reference_forward, lambda m,x:m(x))):
                    expected, actual = baseline(network, inputs), optimized(network, inputs)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)
                    timings = {}
                    # Alternate order to reduce systematic warmup/order effects.
                    for name, function in (("streaming", baseline), ("batched", optimized)) if batch != 32 else (("batched", optimized), ("streaming", baseline)):
                        timings[name] = measure(lambda:function(network, inputs), device)
                    row = dict(device=str(device), batch=batch, scope=scope,
                               streaming_ms=timings["streaming"], batched_ms=timings["batched"],
                               speedup=timings["streaming"]/timings["batched"],
                               max_absolute_error=float((actual-expected).abs().max()))
                    report["measurements"].append(row)
                    print(f"{device} B={batch:3} {scope:7}: {row['streaming_ms']:.3f} -> {row['batched_ms']:.3f} ms ({row['speedup']:.2f}x)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
