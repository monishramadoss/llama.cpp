# Folder-path (sharded) cold tier for `llama_kv_pager`

> Status: **design / approved**. Adds a folder-based, sharded, resumable cold
> tier to the disk-backed KV pager (`src/llama-kv-pager.{h,cpp}`), selected via a
> new `disk_shards` parameter, plus a config-ready CLI surface. The pager's cold
> tier is still developed and unit-tested in isolation; this change does not wire
> it into the live decode path (that remains roadmap item #4 in
> `docs/kv-cache-disk-offload.md`).

## Motivation

Today the pager's cold tier is a single pre-sized backing file opened with
`fopen("w+b")` (`llama-kv-pager.cpp:65-95`, `disk_read_page`/`disk_write_page`).
Two limitations:

1. **One file, one device.** A single file can't spread I/O across multiple
   disks / a RAID set, and a single huge file is awkward to manage.
2. **Scratch only.** `"w+b"` truncates on every run and the file is pre-zeroed
   page-by-page at startup (`O(n_pages · page_bytes)` writes) — there is no way
   to keep a long context's KV across a restart.

This change adds a **folder backend**: pages striped across a configurable number
of shard files inside a directory, with an optional **resume** path so a folder
that already holds valid shards can be reused across runs.

## Design decisions (locked)

| Decision            | Choice                                                        |
|---------------------|---------------------------------------------------------------|
| Target layer        | Pager engine (`llama-kv-pager.{h,cpp}`)                        |
| Folder layout       | **Sharded** — N shard files, pages striped across them        |
| Lifecycle           | **Persistent / resumable** (manifest + validation)            |
| Shard count         | **Configurable** with a sensible default                      |
| Abstraction style   | **Enum + `switch` inside the pager** (no virtual interface)    |
| Scope               | Engine + unit tests + config surface (no live-path wiring)    |

## 1. Mode selection & params

New fields on `llama_kv_pager_params`:

```cpp
// number of shard files to stripe pages across. 0 => single backing FILE at
// disk_path (legacy behavior). >=1 => disk_path is treated as a DIRECTORY and
// pages are striped across this many shard files inside it.
uint32_t disk_shards = 0;

// identifies the model / context / kv-type configuration that produced this
// cache. Required to safely resume an existing folder: if a stored manifest's
// fingerprint or geometry mismatches, the folder is treated as fresh and no
// stale data is surfaced. Empty => resume disabled (folder is scratch).
std::string disk_fingerprint;
```

Internal mode enum drives the I/O `switch`:

```cpp
enum disk_mode {
    DISK_NONE,   // no cold tier (disk_path empty) — unchanged
    DISK_FILE,   // single pre-sized file (disk_shards == 0) — unchanged
    DISK_FOLDER, // sharded files in a directory (disk_shards >= 1)
};
```

Resolution at construction:

- `disk_path` empty → `DISK_NONE`.
- `disk_path` set, `disk_shards == 0` → `DISK_FILE` (today's behavior, refactored
  to flow through the same `switch`).
- `disk_path` set, `disk_shards >= 1` → `DISK_FOLDER`. `disk_path` must name a
  directory (created if absent); error if it exists and is not a directory.

A default shard count is applied by the **caller** (config layer), not the pager:
the pager treats `disk_shards` literally. The config default is `8` (see §4).

## 2. Sharded layout

- **Striping:** `shard = page % n_shards`,
  `offset_in_shard = (page / n_shards) * page_bytes`.
- **Shard sizing:** shard `s` holds the pages `{ p : p % n_shards == s }`; its
  size is `pages_in_shard(s) * page_bytes` where
  `pages_in_shard(s) = (n_pages - s + n_shards - 1) / n_shards` (i.e. the first
  `n_pages % n_shards` shards get one extra page). For simplicity each shard is
  sized to the maximum, `ceil(n_pages / n_shards) * page_bytes`; the trailing
  slack in short shards is never addressed and costs nothing on a sparse file.
- **Sparse creation:** shards are created and sized via `ftruncate` (POSIX) /
  `_chsize_s` (Windows), **not** by writing zeros. Reads of never-written pages
  fall in a file hole and return zero. This avoids the `O(n_pages · page_bytes)`
  startup cost the single-file path pays. (The legacy `DISK_FILE` path keeps its
  current pre-zero behavior to minimize churn / risk.)
- **File naming:** `<folder>/shard-000.bin … shard-NNN.bin`, zero-padded to the
  width needed for `n_shards`.

## 3. Resume (persistent folder)

### Manifest

`<folder>/kv-cache.manifest`, a small fixed-layout file:

```
magic        : "KVPGRMF1" (8 bytes)
version      : uint32
page_bytes   : uint64
n_pages      : uint32
n_shards     : uint32
fp_len       : uint32          // fingerprint byte length
fingerprint  : fp_len bytes
valid_bitmap : ceil(n_pages/8) bytes  // bit p set => page p holds real data
```

### Construction in `DISK_FOLDER` mode

1. If a manifest exists, read it and evaluate `manifest_compatible(...)`
   (**implemented by the user — see §6**). If compatible:
   - open the existing shard files (verify each shard's size ≥ expected),
   - load `valid_bitmap` into the in-memory `page_valid_` bitmap,
   - seed the page table: for each page with its valid bit set, set
     `pages_[p].tier = LLAMA_KV_TIER_DISK` (data is on disk, not yet staged).
2. Otherwise (no manifest, or incompatible): **fresh** — (re)create shard files
   sized as in §2, zero `page_valid_`, write a fresh manifest. Existing files in
   the folder are overwritten only if they are our shard/manifest names; the
   folder itself is never wholesale-deleted.

### Write-back & durability

- `disk_write_page(page, src)` in folder mode writes to the page's shard at its
  offset and sets `page_valid_[page]`.
- `flush()` (existing API) additionally rewrites the manifest's `valid_bitmap`
  (and fsyncs the shards + manifest) so a restart sees a consistent set of valid
  pages. A crash between a page write and the manifest rewrite at worst loses the
  "valid" mark for the newest pages (they re-read as zero), never corrupts older
  data.

### Read of an unwritten page (fresh folder)

`disk_read_page(page, dst)`: if `page_valid_[page]` is unset, `memset(dst, 0,
page_bytes)` and return (no I/O) — same semantics the single-file path gets from
pre-zeroing. If set, read from the shard.

## 4. Config surface (config-ready, not yet live)

The pager's cold tier is **not** wired into live decode yet, so this surface has
no runtime effect; it is landed ahead of wiring, mirroring how `kv_offload_disk`
/ `kv_disk_path` were introduced. Add `kv_disk_shards` symmetrically wherever
`kv_disk_path` already appears:

| File                    | Change                                                        |
|-------------------------|---------------------------------------------------------------|
| `include/llama.h`       | `uint32_t kv_disk_shards;` in `llama_context_params` (+ doc)   |
| `src/llama-context.cpp` | copy through to memory params; default `0` in the defaults blk |
| `src/llama-memory.h`    | `uint32_t kv_disk_shards;`                                     |
| `src/llama-kv-cache.{h,cpp}` | accept + store `kv_disk_shards` (inert, symmetric w/ path)|
| `src/llama-model.cpp`   | pass `params.kv_disk_shards` into the kv-cache ctor            |
| `common/common.h`       | `uint32_t kv_disk_shards = 0;`                                 |
| `common/common.cpp`     | `cparams.kv_disk_shards = params.kv_disk_shards;`              |
| `common/arg.cpp`        | `--kv-disk-shards N` → sets shards + `kv_offload_disk = true`; default applied (8) when a folder path is given with shards unset |

Update `docs/kv-cache-disk-offload.md`'s configuration table with the new field.

## 5. Tests (`tests/test-kv-pager.cpp`)

New cases, alongside the existing single-file coverage (which must keep passing):

1. **Folder round-trip + eviction.** `disk_shards = N`, force pages through
   GPU→RAM→disk and back; verify bytes survive and the expected `shard-*.bin`
   files exist.
2. **Striping correctness.** `n_pages > n_shards` so multiple pages land in each
   shard at distinct offsets; verify no aliasing across the `page % n_shards`
   boundary.
3. **Resume round-trip.** Build pager, write + flush several pages, destroy;
   rebuild on the same folder with a matching fingerprint; assert the written
   pages report `LLAMA_KV_TIER_DISK` and re-acquire with identical bytes.
4. **Resume rejection.** Rebuild with a mismatched fingerprint and with mismatched
   geometry (`page_bytes` / `n_pages` / `n_shards`); assert the folder is treated
   as fresh — pages read back as zero, no stale bytes from the prior run.
5. **Mode selection.** `disk_shards == 0` still selects `DISK_FILE`; all existing
   single-file tests unchanged.
6. **Invalid configs.** `disk_shards >= 1` with empty `disk_path`; `disk_path`
   exists but is a regular file; `n_shards > n_pages` (clamp or error — pick one
   and assert it).

## 6. User-implemented unit: `manifest_compatible(...)`

The resume-safety predicate is a judgment call (which mismatches are fatal vs.
recoverable). It will be scaffolded with the manifest struct, the live params,
and a clear `TODO`, for the user to fill in (~5-8 lines):

```cpp
// Decide whether an existing on-disk manifest may be reused for the current run.
// Returns true only if reusing the folder's shards is safe; false forces a fresh
// (scratch) folder. Geometry mismatches (page_bytes / n_pages / n_shards) must be
// fatal-to-reuse; a supplied fingerprint must match; an empty current fingerprint
// means resume was not requested.
static bool manifest_compatible(const kv_manifest & on_disk,
                                const llama_kv_pager_params & want) {
    // TODO(user): implement the compatibility check.
}
```

Location: top of `src/llama-kv-pager.cpp`, near the other file-scope helpers, with
a TODO marker. The plan will stop at this function for the user's contribution.

## Architecture notes

- The folder logic lives entirely inside `llama_kv_pager`; the public class
  surface gains only the two `params` fields. The page table, LRU, eviction, and
  staging logic are untouched — only the cold-tier I/O and construction change.
- Keeping the legacy `DISK_FILE` path byte-for-byte (pre-zero + single fd) limits
  blast radius; folder mode is additive.
- `madvise`/`MADV_SEQUENTIAL`-style hints are out of scope for the stdio-based
  pager backend (the live mmap cache already does this separately).

## Out of scope (follow-up)

- Wiring the pager cold tier (file or folder) into the live decode path
  (roadmap item #4).
- Async / threaded shard I/O, per-shard placement on distinct mounts.
- Compaction / GC of stale pages within a resumed folder.
