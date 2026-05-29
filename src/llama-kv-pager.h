#pragma once

// Disk-backed, tiered KV-cache pager.
//
// This module implements the storage engine described in the "Disk-Backed
// (RAM + Disk Tiered) Memory" design: a block-paged store whose pages may live
// in one of three tiers:
//
//   - GPU VRAM (hot)  : a small, fixed pool of page slots used for compute
//   - host RAM (warm) : a larger pool that stages data between GPU and disk
//   - disk     (cold) : a pre-sized backing file holding every page
//
// The pager is intentionally decoupled from the attention compute graph so it
// can be developed and unit-tested in isolation (the heavy lifting of wiring
// streamed attention into llama-graph.cpp is layered on top separately).
//
// Tier-to-tier data movement is performed through pluggable copy hooks
// (llama_kv_pager_io). The default hooks use memcpy, which makes the pager
// fully functional and testable on CPU-only systems. A GPU backend can supply
// hooks that wrap ggml_backend_tensor_set / ggml_backend_tensor_get (and an
// async copy stream) so host<->device transfers overlap with compute.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// the tier a logical page currently resides in
enum llama_kv_tier {
    LLAMA_KV_TIER_NONE = 0, // page has never been written (no valid data anywhere)
    LLAMA_KV_TIER_DISK,     // resident only on disk
    LLAMA_KV_TIER_RAM,      // resident in host RAM (and possibly disk)
    LLAMA_KV_TIER_GPU,      // resident in the GPU pool (hot)
};

// pluggable copy hooks used to move a single page (page_bytes large) between
// tiers. dst/src are the addresses returned by the pool allocators below.
// The defaults use memcpy; a GPU backend can override host<->device copies.
struct llama_kv_pager_io {
    // copy host RAM -> GPU slot (stage up)
    std::function<void(void * dst_gpu, const void * src_ram, size_t n)> copy_ram_to_gpu;
    // copy GPU slot -> host RAM (spill down)
    std::function<void(void * dst_ram, const void * src_gpu, size_t n)> copy_gpu_to_ram;

    // construct default (memcpy-based) hooks suitable for CPU execution/testing
    static llama_kv_pager_io make_default();
};

// configuration for the pager
struct llama_kv_pager_params {
    // size of a single page in bytes (must be > 0)
    size_t page_bytes = 0;

    // total number of logical pages backing the full context
    uint32_t n_pages = 0;

    // number of resident slots in each tier pool
    uint32_t gpu_pages = 0; // hot pool capacity (must be >= 1)
    uint32_t ram_pages = 0; // warm pool capacity (0 => no warm tier, GPU<->disk directly)

    // number of pages to prefetch ahead on a sequential acquire (0 => disabled)
    uint32_t prefetch_depth = 0;

    // path to the disk backing file. If empty, the cold tier is disabled and
    // the pager keeps everything in RAM (requires ram_pages >= n_pages) or GPU.
    std::string disk_path;

    // emit verbose diagnostics to stderr
    bool verbose = false;
};

// runtime counters, useful for tests and benchmarking
struct llama_kv_pager_stats {
    uint64_t gpu_hits      = 0; // acquire found page already in GPU
    uint64_t ram_hits      = 0; // acquire staged page up from RAM
    uint64_t disk_reads     = 0; // acquire / stage read a page from disk
    uint64_t disk_writes    = 0; // dirty page written back to disk
    uint64_t gpu_evictions  = 0; // pages evicted out of the GPU pool
    uint64_t ram_evictions  = 0; // pages evicted out of the RAM pool
};

// A block-paged, tiered store for KV-cache pages.
//
// Logical pages are addressed by index in [0, n_pages). Each page may be made
// GPU-resident with acquire_gpu(); the pager transparently stages it up from
// the warm/cold tiers and evicts colder pages to make room, mirroring a
// virtual-memory pager. Pages modified by compute must be marked dirty so they
// are written back before their slot is reused.
class llama_kv_pager {
public:
    llama_kv_pager(const llama_kv_pager_params & params, llama_kv_pager_io io = llama_kv_pager_io::make_default());
    ~llama_kv_pager();

    llama_kv_pager(const llama_kv_pager &) = delete;
    llama_kv_pager & operator=(const llama_kv_pager &) = delete;

    const llama_kv_pager_params & params() const { return params_; }
    const llama_kv_pager_stats  & stats()  const { return stats_; }

    // current tier of a logical page
    llama_kv_tier tier_of(uint32_t page) const;

    // true if the configuration can hold every page resident without ever
    // touching disk (the "single-tier fallback" that matches today's behavior)
    bool is_fully_resident() const;

    // ensure logical page is resident in the GPU pool and return a pointer to
    // its slot. The pointer is valid until the next acquire_gpu()/prefetch()
    // call that could evict it. Updates the LRU clock and triggers prefetch of
    // the following pages when prefetch_depth > 0.
    void * acquire_gpu(uint32_t page);

    // mark a GPU-resident page as modified so it is written back on eviction or
    // flush(). It is an error to mark a page that is not currently GPU-resident.
    void mark_dirty(uint32_t page);

    // hint that page is about to be needed; stage it up to RAM (and optionally
    // GPU) without forcing eviction churn. Safe no-op if already resident.
    void prefetch(uint32_t page, bool to_gpu = false);

    // write all dirty pages down to disk (or RAM if no disk tier). After this
    // call all data is durable on the coldest configured tier.
    void flush();

private:
    struct page_meta {
        llama_kv_tier tier   = LLAMA_KV_TIER_NONE;
        int32_t       slot   = -1;    // index into the tier pool currently holding it (GPU/RAM)
        bool          dirty  = false; // modified since last write-back to disk
        uint64_t      last_use = 0;   // LRU clock value
    };

    struct pool {
        uint32_t              capacity = 0;
        size_t                page_bytes = 0;
        std::vector<uint8_t>  data;          // capacity * page_bytes
        std::vector<int32_t>  slot_page;     // page index resident in each slot, -1 if free

        void   init(uint32_t cap, size_t pb);
        void * slot_ptr(int32_t slot);
        int32_t find_free() const;
    };

    void *  ram_slot_ptr(int32_t slot) { return ram_.slot_ptr(slot); }
    void *  gpu_slot_ptr(int32_t slot) { return gpu_.slot_ptr(slot); }

    // evict the LRU page out of the GPU pool, spilling it to RAM (or disk).
    // returns the freed slot index.
    int32_t evict_one_gpu();
    // evict the LRU page out of the RAM pool, spilling it to disk.
    // returns the freed slot index.
    int32_t evict_one_ram();

    // ensure a page is present in RAM, reading from disk if necessary.
    // returns the RAM slot holding it.
    int32_t ensure_ram(uint32_t page);

    void disk_read_page(uint32_t page, void * dst);
    void disk_write_page(uint32_t page, const void * src);

    int32_t lru_page_in_tier(llama_kv_tier tier) const;

    llama_kv_pager_params params_;
    llama_kv_pager_io     io_;
    llama_kv_pager_stats  stats_;

    pool gpu_;
    pool ram_;

    std::vector<page_meta>      pages_;

    // disk backing for the cold tier; null when no disk tier is configured.
    struct file_deleter { void operator()(std::FILE * f) const { if (f) std::fclose(f); } };
    std::unique_ptr<std::FILE, file_deleter> file_;

    uint64_t clock_ = 0; // monotonically increasing LRU clock
};
