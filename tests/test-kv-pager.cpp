// Unit tests for the disk-backed tiered KV-cache pager (src/llama-kv-pager.*).
//
// These tests exercise the storage engine in isolation on CPU using the
// default (memcpy) copy hooks: page-table consistency, LRU eviction across the
// GPU/RAM/disk tiers, dirty write-back, disk round-trips, prefetching and the
// single-tier fallback.

#include "llama-kv-pager.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

static std::string tmp_path(const char * name) {
    std::string dir;
    const char * env = getenv("TMPDIR");
    dir = env ? env : "/tmp";
    return dir + "/" + name + "-" + std::to_string((uintptr_t) &name) + ".bin";
}

static std::string tmp_dir(const char * name) {
    const char * env = getenv("TMPDIR");
    std::string dir = env ? env : "/tmp";
    return dir + "/" + name + "-" + std::to_string((uintptr_t) &name);
}

static void remove_dir(const std::string & dir) {
    // best-effort cleanup of the shard files + manifest a test created
    for (uint32_t s = 0; s < 64; ++s) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "/shard-%03u.bin", s);
        std::remove((dir + buf).c_str());
    }
    std::remove((dir + "/kv-cache.manifest").c_str());
    rmdir(dir.c_str());
}

// fill a page-sized buffer with a deterministic pattern derived from page index
static std::vector<uint8_t> make_page(size_t page_bytes, uint32_t page) {
    std::vector<uint8_t> v(page_bytes);
    for (size_t i = 0; i < page_bytes; ++i) {
        v[i] = (uint8_t) ((page * 131u + i * 7u) & 0xff);
    }
    return v;
}

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

// Write a distinct pattern into every page, then read them all back. With a
// tiny GPU pool and disk tier this forces repeated eviction + write-back +
// disk reads, validating end-to-end durability of the pager.
static int test_round_trip_with_disk() {
    const size_t   page_bytes = 256;
    const uint32_t n_pages    = 16;

    llama_kv_pager_params p;
    p.page_bytes     = page_bytes;
    p.n_pages        = n_pages;
    p.gpu_pages      = 2;
    p.ram_pages      = 3;
    p.prefetch_depth = 0;
    p.disk_path      = tmp_path("kv-pager-rt");

    llama_kv_pager pager(p);
    CHECK(!pager.is_fully_resident());

    // write phase
    for (uint32_t pg = 0; pg < n_pages; ++pg) {
        void * slot = pager.acquire_gpu(pg);
        CHECK(slot != nullptr);
        CHECK(pager.tier_of(pg) == LLAMA_KV_TIER_GPU);
        const auto data = make_page(page_bytes, pg);
        memcpy(slot, data.data(), page_bytes);
        pager.mark_dirty(pg);
    }

    pager.flush();

    // read phase: every page must read back exactly what was written
    for (uint32_t pg = 0; pg < n_pages; ++pg) {
        const void * slot = pager.acquire_gpu(pg);
        CHECK(slot != nullptr);
        const auto expected = make_page(page_bytes, pg);
        CHECK(memcmp(slot, expected.data(), page_bytes) == 0);
    }

    // we wrote more pages than fit in the resident tiers, so disk must have
    // been used for both writes and reads
    CHECK(pager.stats().disk_writes > 0);
    CHECK(pager.stats().disk_reads  > 0);
    CHECK(pager.stats().gpu_evictions > 0);

    remove(p.disk_path.c_str());
    return 0;
}

// GPU-resident hit should not touch lower tiers and should update LRU.
static int test_gpu_hit() {
    llama_kv_pager_params p;
    p.page_bytes = 64;
    p.n_pages    = 8;
    p.gpu_pages  = 4;
    p.ram_pages  = 2;
    p.disk_path  = tmp_path("kv-pager-hit");

    llama_kv_pager pager(p);

    pager.acquire_gpu(0);
    const uint64_t reads_before = pager.stats().disk_reads;
    const uint64_t hits_before  = pager.stats().gpu_hits;

    // re-acquire the same page: should be a pure GPU hit
    pager.acquire_gpu(0);
    CHECK(pager.stats().gpu_hits == hits_before + 1);
    CHECK(pager.stats().disk_reads == reads_before);
    CHECK(pager.tier_of(0) == LLAMA_KV_TIER_GPU);

    remove(p.disk_path.c_str());
    return 0;
}

// Pages evicted from GPU should land in the RAM (warm) tier when one exists,
// and a subsequent acquire should be served from RAM rather than disk.
static int test_warm_tier_staging() {
    llama_kv_pager_params p;
    p.page_bytes = 128;
    p.n_pages    = 8;
    p.gpu_pages  = 1; // force eviction on every new page
    p.ram_pages  = 4;
    p.disk_path  = tmp_path("kv-pager-warm");

    llama_kv_pager pager(p);

    pager.acquire_gpu(0);
    pager.acquire_gpu(1); // evicts page 0 -> RAM
    CHECK(pager.tier_of(0) == LLAMA_KV_TIER_RAM);

    const uint64_t ram_hits_before = pager.stats().ram_hits;
    pager.acquire_gpu(0); // should be served from RAM (evicting page 1 -> RAM)
    CHECK(pager.stats().ram_hits == ram_hits_before + 1);
    CHECK(pager.tier_of(0) == LLAMA_KV_TIER_GPU);

    remove(p.disk_path.c_str());
    return 0;
}

// No disk tier: the resident pools must hold every page and is_fully_resident
// must report true (the single-tier fallback that matches legacy behavior).
static int test_no_disk_fully_resident() {
    llama_kv_pager_params p;
    p.page_bytes = 32;
    p.n_pages    = 4;
    p.gpu_pages  = 2;
    p.ram_pages  = 2; // 2 + 2 == n_pages
    // no disk_path

    llama_kv_pager pager(p);
    CHECK(pager.is_fully_resident());

    for (uint32_t pg = 0; pg < p.n_pages; ++pg) {
        void * slot = pager.acquire_gpu(pg);
        const auto data = make_page(p.page_bytes, pg);
        memcpy(slot, data.data(), p.page_bytes);
        pager.mark_dirty(pg);
    }
    // never touched disk
    CHECK(pager.stats().disk_writes == 0);
    CHECK(pager.stats().disk_reads  == 0);
    return 0;
}

// Prefetching ahead should warm the following pages into the RAM tier.
static int test_prefetch() {
    llama_kv_pager_params p;
    p.page_bytes     = 64;
    p.n_pages        = 8;
    p.gpu_pages      = 1;
    p.ram_pages      = 4;
    p.prefetch_depth = 2;
    p.disk_path      = tmp_path("kv-pager-pf");

    llama_kv_pager pager(p);

    pager.acquire_gpu(0); // should prefetch pages 1 and 2 into RAM
    CHECK(pager.tier_of(1) == LLAMA_KV_TIER_RAM);
    CHECK(pager.tier_of(2) == LLAMA_KV_TIER_RAM);

    remove(p.disk_path.c_str());
    return 0;
}

// Invalid configurations must be rejected.
static int test_invalid_config() {
    bool threw = false;
    try {
        llama_kv_pager_params p;
        p.page_bytes = 0; // invalid
        p.n_pages    = 4;
        p.gpu_pages  = 1;
        llama_kv_pager pager(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        // no disk and pools too small to hold all pages
        llama_kv_pager_params p;
        p.page_bytes = 16;
        p.n_pages    = 10;
        p.gpu_pages  = 1;
        p.ram_pages  = 1;
        llama_kv_pager pager(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
    return 0;
}

// flush() must write dirty pages back to disk even when they have been evicted
// to the RAM (warm) tier, and mark_dirty() must reject pages that are not
// GPU-resident.
static int test_flush_from_ram_and_mark_dirty_errors() {
    const size_t   page_bytes = 64;
    const uint32_t n_pages    = 6;

    llama_kv_pager_params p;
    p.page_bytes = page_bytes;
    p.n_pages    = n_pages;
    p.gpu_pages  = 1; // force the written page to be evicted to RAM
    p.ram_pages  = 4;
    p.disk_path  = tmp_path("kv-pager-flush");

    llama_kv_pager pager(p);

    // write page 0, then touch page 1 so page 0 is evicted into the RAM tier
    void * slot0 = pager.acquire_gpu(0);
    const auto data0 = make_page(page_bytes, 0);
    memcpy(slot0, data0.data(), page_bytes);
    pager.mark_dirty(0);

    pager.acquire_gpu(1);      // evicts dirty page 0 down to RAM
    CHECK(pager.tier_of(0) == LLAMA_KV_TIER_RAM);

    // mark_dirty on a non-GPU-resident page must throw
    bool threw = false;
    try {
        pager.mark_dirty(0);   // page 0 is in RAM, not GPU
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);

    // flush must write the dirty RAM-resident page 0 back to disk
    const uint64_t writes_before = pager.stats().disk_writes;
    pager.flush();
    CHECK(pager.stats().disk_writes > writes_before);

    // and the data must read back correctly
    const void * slot = pager.acquire_gpu(0);
    CHECK(memcmp(slot, data0.data(), page_bytes) == 0);

    remove(p.disk_path.c_str());
    return 0;
}

// Folder (sharded) cold tier: write a distinct pattern into every page with a
// tiny GPU pool so pages are forced out to the shard files, then read them all
// back. Verifies striping + eviction + write-back + disk reads across shards.
static int test_folder_round_trip() {
    const size_t   page_bytes = 256;
    const uint32_t n_pages    = 20;

    const std::string dir = tmp_dir("kv-pager-folder-rt");
    remove_dir(dir); // start clean

    llama_kv_pager_params p;
    p.page_bytes  = page_bytes;
    p.n_pages     = n_pages;
    p.gpu_pages   = 2;
    p.ram_pages   = 3;
    p.disk_path   = dir;
    p.disk_shards = 4; // -> DISK_FOLDER, 4 shard files

    llama_kv_pager pager(p);
    CHECK(!pager.is_fully_resident());

    for (uint32_t pg = 0; pg < n_pages; ++pg) {
        void * slot = pager.acquire_gpu(pg);
        CHECK(slot != nullptr);
        const auto data = make_page(page_bytes, pg);
        memcpy(slot, data.data(), page_bytes);
        pager.mark_dirty(pg);
    }
    pager.flush();

    for (uint32_t pg = 0; pg < n_pages; ++pg) {
        const void * slot = pager.acquire_gpu(pg);
        CHECK(slot != nullptr);
        const auto expected = make_page(page_bytes, pg);
        CHECK(memcmp(slot, expected.data(), page_bytes) == 0);
    }

    CHECK(pager.stats().disk_writes  > 0);
    CHECK(pager.stats().disk_reads   > 0);
    CHECK(pager.stats().gpu_evictions > 0);

    // the four shard files must exist
    for (uint32_t s = 0; s < 4; ++s) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "/shard-%03u.bin", s);
        std::FILE * f = std::fopen((dir + buf).c_str(), "rb");
        CHECK(f != nullptr);
        std::fclose(f);
    }

    remove_dir(dir);
    return 0;
}

// A page that has never been written in a fresh folder must read back as zero
// and must not trigger a shard read (the valid bitmap short-circuits the I/O).
static int test_folder_unwritten_reads_zero() {
    const size_t   page_bytes = 128;
    const uint32_t n_pages    = 12;

    const std::string dir = tmp_dir("kv-pager-folder-zero");
    remove_dir(dir);

    llama_kv_pager_params p;
    p.page_bytes  = page_bytes;
    p.n_pages     = n_pages;
    p.gpu_pages   = 2;
    p.ram_pages   = 2;
    p.disk_path   = dir;
    p.disk_shards = 3;

    llama_kv_pager pager(p);

    const uint64_t reads_before = pager.stats().disk_reads;
    const void * slot = pager.acquire_gpu(7); // never written
    CHECK(slot != nullptr);

    std::vector<uint8_t> zero(page_bytes, 0);
    CHECK(memcmp(slot, zero.data(), page_bytes) == 0);
    CHECK(pager.stats().disk_reads == reads_before); // no shard read happened

    remove_dir(dir);
    return 0;
}

// Build a folder cache, write + flush some pages, destroy it, then rebuild on
// the same folder with a matching fingerprint. The written pages must come back
// as DISK-resident with their exact bytes (resume across "process restarts").
static int test_folder_resume_round_trip() {
    const size_t   page_bytes = 192;
    const uint32_t n_pages    = 16;
    const uint32_t written[]  = {0, 5, 9, 15};

    const std::string dir = tmp_dir("kv-pager-folder-resume");
    remove_dir(dir);

    {
        llama_kv_pager_params p;
        p.page_bytes       = page_bytes;
        p.n_pages          = n_pages;
        p.gpu_pages        = 2;
        p.ram_pages        = 2;
        p.disk_path        = dir;
        p.disk_shards      = 4;
        p.disk_fingerprint = "model-A/ctx-16/f16";

        llama_kv_pager pager(p);
        for (uint32_t pg : written) {
            void * slot = pager.acquire_gpu(pg);
            const auto data = make_page(page_bytes, pg);
            memcpy(slot, data.data(), page_bytes);
            pager.mark_dirty(pg);
        }
        pager.flush(); // persists data + manifest (valid bitmap)
    } // first pager destroyed -> shards closed

    {
        llama_kv_pager_params p;
        p.page_bytes       = page_bytes;
        p.n_pages          = n_pages;
        p.gpu_pages        = 2;
        p.ram_pages        = 2;
        p.disk_path        = dir;
        p.disk_shards      = 4;
        p.disk_fingerprint = "model-A/ctx-16/f16"; // matches -> resume

        llama_kv_pager pager(p);
        for (uint32_t pg : written) {
            CHECK(pager.tier_of(pg) == LLAMA_KV_TIER_DISK); // seeded from manifest
        }
        for (uint32_t pg : written) {
            const void * slot = pager.acquire_gpu(pg);
            const auto expected = make_page(page_bytes, pg);
            CHECK(memcmp(slot, expected.data(), page_bytes) == 0);
        }
    }

    remove_dir(dir);
    return 0;
}

// Resume must be REJECTED when the fingerprint or geometry differs, and when no
// fingerprint is supplied. In every rejected case the folder is treated as fresh
// and previously written pages must read back as zero (no stale data surfaced).
static int test_folder_resume_rejected_on_mismatch() {
    const size_t   page_bytes = 128;
    const uint32_t n_pages    = 12;
    const uint32_t probe      = 4;

    const std::string dir = tmp_dir("kv-pager-folder-reject");

    auto seed_folder = [&](const std::string & fingerprint) {
        remove_dir(dir);
        llama_kv_pager_params p;
        p.page_bytes       = page_bytes;
        p.n_pages          = n_pages;
        p.gpu_pages        = 2;
        p.ram_pages        = 2;
        p.disk_path        = dir;
        p.disk_shards      = 3;
        p.disk_fingerprint = fingerprint;
        llama_kv_pager pager(p);
        void * slot = pager.acquire_gpu(probe);
        const auto data = make_page(page_bytes, probe);
        memcpy(slot, data.data(), page_bytes);
        pager.mark_dirty(probe);
        pager.flush();
    };

    auto reopen_expects_fresh = [&](const std::string & fingerprint, uint32_t shards) {
        llama_kv_pager_params p;
        p.page_bytes       = page_bytes;
        p.n_pages          = n_pages;
        p.gpu_pages        = 2;
        p.ram_pages        = 2;
        p.disk_path        = dir;
        p.disk_shards      = shards;
        p.disk_fingerprint = fingerprint;
        llama_kv_pager pager(p);
        // rejected resume => page is NOT seeded as DISK and reads back zero
        CHECK(pager.tier_of(probe) == LLAMA_KV_TIER_NONE);
        const void * slot = pager.acquire_gpu(probe);
        std::vector<uint8_t> zero(page_bytes, 0);
        CHECK(memcmp(slot, zero.data(), page_bytes) == 0);
        return 0;
    };

    // 1) different fingerprint -> fresh
    seed_folder("model-A");
    if (reopen_expects_fresh("model-B", 3)) return 1;

    // 2) different shard count (geometry) -> fresh
    seed_folder("model-A");
    if (reopen_expects_fresh("model-A", 4)) return 1;

    // 3) empty fingerprint -> resume not requested -> fresh
    seed_folder("model-A");
    if (reopen_expects_fresh("", 3)) return 1;

    remove_dir(dir);
    return 0;
}

// Folder-mode misconfigurations must be rejected, and n_shards > n_pages must be
// clamped (so empty shard files are never created).
static int test_folder_invalid_config() {
    // disk_shards >= 1 but no path -> error
    bool threw = false;
    try {
        llama_kv_pager_params p;
        p.page_bytes  = 64;
        p.n_pages     = 4;
        p.gpu_pages   = 1;
        p.ram_pages   = 1;
        p.disk_shards = 2; // folder mode requested without a path
        llama_kv_pager pager(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);

    // disk_path points at a regular file, not a directory -> error
    threw = false;
    const std::string file = tmp_path("kv-pager-notadir");
    {
        std::FILE * f = std::fopen(file.c_str(), "wb");
        CHECK(f != nullptr);
        std::fclose(f);
    }
    try {
        llama_kv_pager_params p;
        p.page_bytes  = 64;
        p.n_pages     = 4;
        p.gpu_pages   = 1;
        p.ram_pages   = 1;
        p.disk_path   = file;
        p.disk_shards = 2;
        llama_kv_pager pager(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
    std::remove(file.c_str());

    // n_shards > n_pages -> clamped to n_pages: exactly n_pages shard files exist,
    // and shard index n_pages (the unclamped count) does not.
    const std::string dir = tmp_dir("kv-pager-clamp");
    remove_dir(dir);
    {
        llama_kv_pager_params p;
        p.page_bytes  = 64;
        p.n_pages     = 3;
        p.gpu_pages   = 1;
        p.ram_pages   = 1;
        p.disk_path   = dir;
        p.disk_shards = 8; // > n_pages -> clamp to 3
        llama_kv_pager pager(p);

        for (uint32_t s = 0; s < 3; ++s) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "/shard-%03u.bin", s);
            std::FILE * f = std::fopen((dir + buf).c_str(), "rb");
            CHECK(f != nullptr);
            std::fclose(f);
        }
        std::FILE * extra = std::fopen((dir + "/shard-003.bin").c_str(), "rb");
        CHECK(extra == nullptr); // shard 3 must not exist (clamped)
    }
    remove_dir(dir);
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_round_trip_with_disk();
    rc |= test_folder_round_trip();
    rc |= test_folder_unwritten_reads_zero();
    rc |= test_folder_resume_round_trip();
    rc |= test_folder_resume_rejected_on_mismatch();
    rc |= test_folder_invalid_config();
    rc |= test_gpu_hit();
    rc |= test_warm_tier_staging();
    rc |= test_no_disk_fully_resident();
    rc |= test_prefetch();
    rc |= test_invalid_config();
    rc |= test_flush_from_ram_and_mark_dirty_errors();

    if (rc == 0) {
        printf("test-kv-pager: all tests passed\n");
    }
    return rc;
}
