# Disk-Backed (RAM + Disk Tiered) KV Cache

> Status: **experimental / foundational**. This document describes the tiered
> KV-cache *storage engine* (`src/llama-kv-pager.{h,cpp}`) and the configuration
> surface that exposes it. Wiring the pager into the live attention compute path
> (streaming / online-softmax attention) is intentionally a separate, follow-up
> step — see [Scope](#scope-and-follow-up-work) below.

## Motivation

For very long contexts (e.g. 1M tokens) the dominant memory cost is the KV
cache, not the model weights. The per-token cost is approximately:

```
bytes/token ≈ 2 (K+V) × n_layer × n_embd_kv × sizeof(type)
```

For a typical 7–8B model this is on the order of hundreds of GB at 1M tokens —
far beyond a single GPU's VRAM and often beyond host RAM as well. Today
llama.cpp allocates the entire KV cache up front in a single backend buffer with
no paging or eviction.

The goal is a **3-tier paged store** so a context far larger than VRAM can be
served from a single GPU by staging data through host RAM and disk:

| Tier        | Role                       | Backing                          |
|-------------|----------------------------|----------------------------------|
| GPU (hot)   | active compute working set | fixed pool of page slots         |
| RAM (warm)  | staging between GPU & disk | larger pool of page slots        |
| Disk (cold) | source of truth, full size | pre-sized backing file           |

## The pager (`llama_kv_pager`)

`src/llama-kv-pager.{h,cpp}` implements a block-paged store, decoupled from the
attention graph so it can be developed and unit-tested in isolation. Key
properties:

- **Fixed-size pages.** The KV cache is split along the token dimension into
  pages. Each logical page lives in exactly one tier at a time.
- **Page table.** Per-page metadata tracks `{tier, slot, dirty, last_use}`,
  mirroring a virtual-memory pager.
- **Demand staging.** `acquire_gpu(page)` makes a page GPU-resident, staging it
  up from RAM, or from disk via RAM, as needed.
- **LRU eviction + write-back.** When a pool is full the least-recently-used
  page is spilled down a tier; dirty pages are written back before their slot is
  reused (`mark_dirty`, `flush`).
- **Prefetch.** Sequential acquires optionally prefetch the following pages into
  the warm tier (`prefetch_depth`).
- **Pluggable copy hooks** (`llama_kv_pager_io`). Host↔device transfers go
  through hooks that default to `memcpy` (so the engine is fully functional and
  testable on CPU). A GPU backend can supply hooks wrapping
  `ggml_backend_tensor_set` / `ggml_backend_tensor_get` and an async copy stream
  so transfers overlap with compute.
- **Single-tier fallback.** If the resident pools can hold every page
  (`is_fully_resident()`), no disk I/O occurs and behavior matches the
  non-tiered cache.

Unit tests live in `tests/test-kv-pager.cpp` and cover eviction across all
tiers, dirty write-back, disk round-trips, page-table consistency, prefetching,
the all-resident fallback, and invalid configurations.

## Configuration

The following experimental fields were added to `llama_context_params` (and
plumbed through to `llama_memory_params`). When `kv_offload_disk` is `false`
(the default) all of these are ignored and the KV cache behaves exactly as
before:

| Field               | Meaning                                             |
|---------------------|-----------------------------------------------------|
| `kv_offload_disk`   | enable the RAM/disk tiered pager                    |
| `kv_disk_path`      | path for the cold (disk) tier                       |
| `kv_page_tokens`    | page size in tokens (0 ⇒ implementation default)    |
| `kv_vram_pages`     | GPU (hot) pool capacity in pages (0 ⇒ default)      |
| `kv_ram_pages`      | host RAM (warm) pool capacity in pages (0 ⇒ default)|
| `kv_prefetch_depth` | pages to prefetch ahead (0 ⇒ disabled)              |

For Ollama (which wraps llama.cpp), no algorithmic change is required: it only
needs to plumb these as runner options that map onto the parameters above.

## Scope and follow-up work

This change lands the **storage engine and configuration surface** only. The
remaining steps from the design, in order, are:

1. Page-based addressing inside the existing KV cache (behavior-identical
   refactor, verify parity). ← partially enabled by this engine
2. Wire the warm (RAM) tier + pinned staging + async H2D/D2H into the live KV
   cache.
3. **Streaming / chunked online-softmax attention** in the graph builder so the
   GPU working set is bounded regardless of context length — this is the key
   piece that makes dense 1M-token attention feasible on bounded VRAM.
4. Disk cold tier in the live path (background I/O thread, prefetch, write-back).
5. Plumb the parameters through the server and Ollama options. ← config surface
   is ready
6. Policy tuning (page size, pool sizes, prefetch depth) + benchmarks.

### Risks / trade-offs

- **Bandwidth wall.** Dense attention reads the whole cache each decode step, so
  throughput is ultimately bounded by PCIe and disk bandwidth. Disk tiering
  makes huge contexts *possible*, not necessarily fast; sliding-window / sparse
  models benefit most.
- **Numerical care.** Streaming softmax must track a running max for stability.
- **Concurrency.** Overlapping compute, H2D/D2H copies, and disk I/O requires
  careful event/stream management to avoid stalls and data races on shared
  pages.
