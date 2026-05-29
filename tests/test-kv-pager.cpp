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

static std::string tmp_path(const char * name) {
    std::string dir;
    const char * env = getenv("TMPDIR");
    dir = env ? env : "/tmp";
    return dir + "/" + name + "-" + std::to_string((uintptr_t) &name) + ".bin";
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

int main() {
    int rc = 0;
    rc |= test_round_trip_with_disk();
    rc |= test_gpu_hit();
    rc |= test_warm_tier_staging();
    rc |= test_no_disk_fully_resident();
    rc |= test_prefetch();
    rc |= test_invalid_config();

    if (rc == 0) {
        printf("test-kv-pager: all tests passed\n");
    }
    return rc;
}
