# MiniMax Stream Interval Harness Note

Date: 2026-05-17

## Intent

Allow the MiniMax quality harness to pass vLLM's `stream_interval` constructor
option through `LLM(...)` so scheduler cadence experiments can be exact-token
gated before benchmark runs.

## Harness Option

```bash
--stream-interval 2048
```

## Implementation Sketch

File patched:

- `scripts/run-vllm-minimax-quality-check.py`

Exact patch:

- `patches/minimax-quality-harness-speculative-stream-interval-20260517.patch`

The patch adds a nullable integer CLI flag:

```python
parser.add_argument(
    "--stream-interval",
    type=int,
    default=None,
    help=(
        "Optional vLLM scheduler stream interval. Larger values reduce "
        "host output frequency without changing sampling."
    ),
)
```

When set, the harness passes it through to vLLM:

```python
if args.stream_interval is not None:
    llm_kwargs["stream_interval"] = args.stream_interval
```

The runtime JSON also records the tested value.

The same local harness delta also preserves the already-tested
`--speculative-config` pass-through so future speculative probes can be gated
through the same quality script before any benchmark is trusted.

## Validation

Syntax check:

```bash
python3 -m py_compile \
  /home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py
```

Quality canary for `--stream-interval 2048`:

- raw145 n64 exact hash passed
- hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Benchmark:

- adjacent control: `61.894320` output tok/s, `82.525760` total tok/s
- stream interval 2048: `61.404979` output tok/s, `81.873305` total tok/s

## Decision

Keep the harness support because it is useful for future scheduler experiments.
Do not promote `--stream-interval 2048` as a throughput improvement on this
recipe.
