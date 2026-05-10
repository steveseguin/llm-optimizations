# Fast NVMe Model Placement, 2026-05-10

The new PCIe 5.0 NVMe is mounted at `/mnt/fast-ai` as ext4 with `noatime`.

Model placement updates:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound` remains at
  `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`.
- `unsloth/MiniMax-M2.7-GGUF` `UD-IQ4_XS` was copied from
  `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf` to
  `/mnt/fast-ai/llm-models/minimax-m2.7-ud-iq4_xs-gguf`.
- Every copied GGUF shard was verified with `cmp -s` before the original root
  copy was removed.
- The original path is preserved as a symlink:
  `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf ->
  /mnt/fast-ai/llm-models/minimax-m2.7-ud-iq4_xs-gguf`.

Space after cleanup:

- `/`: `285G` used, `149G` free, `66%`.
- `/mnt/fast-ai`: `227G` used, `643G` free, `27%`.

This changes load/staging behavior only. It is not a decode-throughput
optimization and should not be counted as a benchmark improvement.
