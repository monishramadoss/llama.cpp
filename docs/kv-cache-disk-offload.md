# Disk-Backed (RAM + Disk Tiered) KV Cache

> Status: **experimental / foundational**. This document describes the tiered
> KV-cache *storage engine* (`src/llama-kv-pager.{h,cpp}`), the numerically
> stable *streaming attention* primitive (`src/llama-attn-stream.{h,cpp}`) it
> feeds, the ggml *graph builder* that runs that primitive in the live compute
> path (`src/llama-attn-stream-graph.{h,cpp}`), and the configuration surface
> that exposes them. The warm/cold storage *tiers* are still a follow-up step —
> see [Scope](#scope-and-follow-up-work) below.

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

## Streaming / online-softmax attention (`llama_attn_stream`)

`src/llama-attn-stream.{h,cpp}` implements the compute-side counterpart to the
pager: numerically stable scaled dot-product attention that consumes the
keys/values **one chunk (page) at a time** instead of materialising the full
attention matrix. This is what keeps the GPU working set bounded regardless of
context length — the pager makes one page of K/V resident, and this primitive
folds that page into a running softmax before the page can be evicted.

It uses the FlashAttention-style online-softmax recurrence, tracking three
running quantities per query and rescaling by the change in the running maximum
for stability:

```
m   = running max of the scores seen so far   (init -inf)
l   = running sum of exp(s_j - m)             (init 0)
acc = running sum of exp(s_j - m) * v_j       (init 0)
out = acc / l                                 (after all chunks)
```

The result is identical (up to floating-point rounding) to a one-shot softmax
but never touches more than one chunk of K/V at once. The API is intentionally
decoupled from the ggml graph so it can be unit-tested on CPU:

- `begin(q)` binds the query matrix and resets the accumulators.
- `update(k_chunk, v_chunk, n_keys, mask)` folds one chunk in; an optional
  additive mask supports causal / sparse attention.
- `finish(out)` writes the normalised output; rows that saw no unmasked key
  produce zeros rather than NaNs.

`llama_attn_reference()` is an independent one-shot oracle used by the tests.
Unit tests live in `tests/test-attn-stream.cpp` and cover parity with the
oracle, invariance to chunk size, numerical stability for very large logits,
causal and fully-masked rows, the empty stream, instance reuse, and invalid
configurations.

## Wiring streaming attention into the graph (`llama_attn_stream_build_graph`)

`src/llama-attn-stream-graph.{h,cpp}` lifts the streaming idea into the live ggml
compute graph. `llama_attn_stream_build_graph()` builds a subgraph that computes
attention while folding the KV in fixed-size chunks along the token dimension and
accumulating the partial softmax numerator/denominator across chunks. The
largest score intermediate it ever materialises is `[chunk, n_query, ...]`
instead of the full `[n_kv, n_query, ...]` matrix, so the attention working set
stays bounded regardless of context length. It uses only public ggml ops, so it
runs on any backend and is unit-tested on the CPU backend
(`tests/test-attn-stream-graph.cpp`) against a dense reference for several chunk
sizes, with and without causal masks.

It is hooked into `llm_graph_context::build_attn_mha` as an opt-in fast path. The
path is enabled by two `llama_cparams` fields, populated from the experimental
KV-offload config:

| cparams field       | source                | meaning                              |
|---------------------|-----------------------|--------------------------------------|
| `attn_streaming`    | `kv_offload_disk`     | use the chunked attention path       |
| `attn_chunk_tokens` | `kv_page_tokens`      | KV tokens folded per chunk (page)    |

When `attn_streaming` is set and the case is the plain softmax case (no KQ bias,
sinks, MLA, soft-capping, ALiBi, or Grok logit scaling) the streaming path is
used **in preference to** flash attention; anything outside that set falls back
to the existing flash / dense paths, so default behavior is unchanged when the
feature is off. `llama_context::graph_max_nodes()` is bumped accordingly so the
extra per-chunk nodes fit in the reserved graph.

`tests/test-attn-stream-e2e.cpp` builds a tiny random LLAMA model and verifies
that decoding with the streaming path enabled (and a small chunk size) matches
the dense path to floating-point rounding.

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

This change lands the **storage engine, streaming-attention primitive, and
configuration surface**. The remaining steps from the design, in order, are:

1. Page-based addressing inside the existing KV cache (behavior-identical
   refactor, verify parity). ← partially enabled by this engine
2. Wire the warm (RAM) tier + pinned staging + async H2D/D2H into the live KV
   cache.
3. Wire the streaming / chunked online-softmax attention primitive
   (`llama_attn_stream`) into the ggml graph builder so the GPU working set is
   bounded regardless of context length. ← **done**: see
   `llama_attn_stream_build_graph` and its integration into `build_attn_mha`
   ("Wiring streaming attention into the graph" above).
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
