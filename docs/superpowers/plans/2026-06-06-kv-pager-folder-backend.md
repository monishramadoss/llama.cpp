# KV Pager Folder-Path (Sharded) Cold Tier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a folder-based, sharded, resumable cold tier to `llama_kv_pager`, selected by a new `disk_shards` parameter, plus a config-ready `kv_disk_shards` CLI surface.

**Architecture:** The pager keeps its single-file path unchanged and gains a third internal mode (`DISK_FOLDER`) chosen when `disk_shards >= 1`. Pages are striped across N shard files inside a directory (`shard = page % n_shards`, `offset = (page / n_shards) * page_bytes`). A `kv-cache.manifest` records geometry + a fingerprint + a valid-page bitmap so an existing folder can be reused across runs. The cold tier is still developed/tested in isolation — no live-decode wiring.

**Tech Stack:** C++17, stdio (`FILE*`) + POSIX/`_WIN32` file ops (`ftruncate`/`_chsize_s`, `mkdir`/`_mkdir`), CMake/CTest, the existing hand-rolled `CHECK` test harness in `tests/test-kv-pager.cpp`.

**Spec:** `docs/superpowers/specs/2026-06-06-kv-pager-folder-backend-design.md`

---

## File Structure

- **Modify** `src/llama-kv-pager.h` — new params (`disk_shards`, `disk_fingerprint`); private mode enum, members, helper + manifest declarations.
- **Modify** `src/llama-kv-pager.cpp` — portable fs helpers; folder construction/resume; sharded `disk_read_page`/`disk_write_page`; manifest read/write; eviction tier-fix; `flush()` manifest rewrite; `manifest_compatible` (user-implemented in Task 4).
- **Modify** `tests/test-kv-pager.cpp` — folder round-trip, unwritten-reads-zero, resume, resume-rejection, invalid/edge configs; register in `main()`.
- **Modify (config surface, inert):** `include/llama.h`, `src/llama-memory.h`, `src/llama-context.cpp` (×2), `src/llama-model.cpp`, `src/llama-kv-cache.h` (×2), `src/llama-kv-cache.cpp`, `common/common.h`, `common/common.cpp`, `common/arg.cpp`.
- **Modify (docs):** `docs/kv-cache-disk-offload.md` — config table + folder/resume note.

---

## Task 0: Establish a green build baseline

**Files:** none (build only).

- [ ] **Step 1: Configure the build (first time only)**

Run: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`
Expected: configuration succeeds, ends with `-- Build files have been written to: .../build`.

- [ ] **Step 2: Build and run the existing pager test**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed`

This confirms the starting point is green before any changes. Do not commit.

---

## Task 1: Folder mode — params, construction, sharded round-trip

**Files:**
- Modify: `src/llama-kv-pager.h`
- Modify: `src/llama-kv-pager.cpp`
- Test: `tests/test-kv-pager.cpp`

- [ ] **Step 1: Add the folder round-trip test**

In `tests/test-kv-pager.cpp`, add a temp-directory helper next to `tmp_path` (after line 22) and a directory-cleanup helper:

```cpp
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
```

Add `#include <unistd.h>` near the top includes (for `rmdir`). Then add this test function (before `main`):

```cpp
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
```

Register it in `main()` — add after the `test_round_trip_with_disk()` line:

```cpp
    rc |= test_folder_round_trip();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --target test-kv-pager -j 2>&1 | tail -20`
Expected: **compile error** — `llama_kv_pager_params` has no member `disk_shards`.

- [ ] **Step 3: Add the header declarations**

In `src/llama-kv-pager.h`, inside `struct llama_kv_pager_params`, immediately after the `disk_path` field (line 69), add:

```cpp
    // number of shard files to stripe pages across. 0 => single backing file at
    // disk_path (legacy). >=1 => disk_path is a directory and pages are striped
    // across this many shard files inside it (clamped to n_pages).
    uint32_t disk_shards = 0;

    // identifies the model/context/kv configuration that produced this cache.
    // Required to safely resume an existing folder: on mismatch the folder is
    // treated as fresh. Empty => resume disabled (folder treated as scratch).
    std::string disk_fingerprint;
```

In the `class llama_kv_pager` **private** section, replace the disk-backing comment + `file_` member (lines 175-177) with the extended state and declarations:

```cpp
    // cold-tier mode
    enum disk_mode { DISK_NONE, DISK_FILE, DISK_FOLDER };

    // on-disk manifest describing a folder cold tier (used to resume a folder)
    struct kv_manifest {
        uint32_t             version    = 0;
        uint64_t             page_bytes = 0;
        uint32_t             n_pages    = 0;
        uint32_t             n_shards   = 0;
        std::string          fingerprint;
        std::vector<uint8_t> valid_bitmap; // bit p set => page p holds real data
    };

    // folder-mode helpers
    uint32_t    shard_of(uint32_t page)     const { return page % n_shards_; }
    size_t      shard_offset(uint32_t page) const { return (size_t)(page / n_shards_) * params_.page_bytes; }
    std::string shard_path(uint32_t shard)  const;
    bool        page_is_valid(uint32_t page) const;
    void        set_page_valid(uint32_t page);
    void        open_or_create_folder();      // ctor helper for DISK_FOLDER
    bool        read_manifest(kv_manifest & out) const;
    void        write_manifest();
    static bool manifest_compatible(const kv_manifest & on_disk,
                                    const llama_kv_pager_params & want,
                                    uint32_t want_shards);

    // disk backing for the cold tier
    struct file_deleter { void operator()(std::FILE * f) const { if (f) std::fclose(f); } };
    disk_mode disk_mode_ = DISK_NONE;
    std::unique_ptr<std::FILE, file_deleter> file_;   // DISK_FILE
    uint32_t  n_shards_ = 0;                           // DISK_FOLDER
    std::string folder_path_;                          // DISK_FOLDER (no trailing '/')
    std::vector<std::unique_ptr<std::FILE, file_deleter>> shards_; // DISK_FOLDER
    std::vector<uint8_t> page_valid_;                  // DISK_FOLDER valid bitmap
```

(Leave the existing `clock_` member and everything else as-is. The `file_deleter` struct moves up with this block — delete its old standalone definition at lines 176-177 so it is not declared twice.)

- [ ] **Step 4: Add portable fs helpers at the top of the .cpp**

In `src/llama-kv-pager.cpp`, after the existing `#include` block and the `llama_kv_pager_fseek` macro (after line 15), add:

```cpp
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#  include <direct.h>
#  include <io.h>
#  include <sys/stat.h>
static int  kv_mkdir(const char * p)                 { return _mkdir(p); }
static int  kv_ftruncate(std::FILE * f, long long n) { return _chsize_s(_fileno(f), n); }
static bool kv_isdir(const char * p)  { struct _stat st; return _stat(p, &st) == 0 && (st.st_mode & _S_IFDIR); }
static bool kv_exists(const char * p) { struct _stat st; return _stat(p, &st) == 0; }
#else
#  include <sys/stat.h>
#  include <unistd.h>
static int  kv_mkdir(const char * p)                 { return mkdir(p, 0700); }
static int  kv_ftruncate(std::FILE * f, long long n) { return ftruncate(fileno(f), (off_t) n); }
static bool kv_isdir(const char * p)  { struct stat st; return stat(p, &st) == 0 && S_ISDIR(st.st_mode); }
static bool kv_exists(const char * p) { struct stat st; return stat(p, &st) == 0; }
#endif

static const char     KV_MANIFEST_MAGIC[8] = {'K','V','P','G','R','M','F','1'};
static const uint32_t KV_MANIFEST_VERSION  = 1;
```

- [ ] **Step 5: Route construction through the mode enum**

In `src/llama-kv-pager.cpp`, replace the disk-setup block in the constructor (lines 65-88, the `if (!params_.disk_path.empty()) { ... } else { ... }`) with:

```cpp
    if (params_.disk_path.empty()) {
        disk_mode_ = DISK_NONE;
        // without a disk tier every page must fit in the resident tiers
        if (params_.ram_pages + params_.gpu_pages < params_.n_pages) {
            throw std::runtime_error(
                "llama_kv_pager: no disk tier and resident pools too small to hold all pages");
        }
    } else if (params_.disk_shards >= 1) {
        disk_mode_ = DISK_FOLDER;
        open_or_create_folder();
    } else {
        disk_mode_ = DISK_FILE;
        // open the backing file for read+write and pre-size it so page offsets
        // are valid up front. "w+b" truncates any existing file (fresh scratch).
        std::FILE * fp = std::fopen(params_.disk_path.c_str(), "w+b");
        if (!fp) {
            throw std::runtime_error("llama_kv_pager: failed to open disk backing file: " + params_.disk_path);
        }
        file_.reset(fp);
        std::vector<uint8_t> zero(params_.page_bytes, 0);
        for (uint32_t i = 0; i < params_.n_pages; ++i) {
            if (std::fwrite(zero.data(), 1, zero.size(), file_.get()) != zero.size()) {
                throw std::runtime_error("llama_kv_pager: failed to pre-size disk backing file");
            }
        }
        std::fflush(file_.get());
    }
```

- [ ] **Step 6: Implement the folder helpers + construction**

In `src/llama-kv-pager.cpp`, add these definitions after the constructor (after line 96, before `~llama_kv_pager`):

```cpp
std::string llama_kv_pager::shard_path(uint32_t shard) const {
    char buf[32];
    std::snprintf(buf, sizeof buf, "/shard-%03u.bin", shard);
    return folder_path_ + buf;
}

bool llama_kv_pager::page_is_valid(uint32_t page) const {
    return (page_valid_[page >> 3] >> (page & 7)) & 1u;
}

void llama_kv_pager::set_page_valid(uint32_t page) {
    page_valid_[page >> 3] |= (uint8_t) (1u << (page & 7));
}

void llama_kv_pager::open_or_create_folder() {
    folder_path_ = params_.disk_path;
    while (!folder_path_.empty() && folder_path_.back() == '/') {
        folder_path_.pop_back();
    }

    if (kv_exists(folder_path_.c_str())) {
        if (!kv_isdir(folder_path_.c_str())) {
            throw std::runtime_error("llama_kv_pager: disk_path exists and is not a directory: " + folder_path_);
        }
    } else if (kv_mkdir(folder_path_.c_str()) != 0) {
        throw std::runtime_error("llama_kv_pager: failed to create folder: " + folder_path_);
    }

    // clamp shard count so empty shards are never created
    n_shards_ = params_.disk_shards;
    if (n_shards_ > params_.n_pages) {
        n_shards_ = params_.n_pages;
    }

    const size_t bitmap_bytes = (params_.n_pages + 7) / 8;
    page_valid_.assign(bitmap_bytes, 0);

    const uint32_t  pages_per_shard = (params_.n_pages + n_shards_ - 1) / n_shards_;
    const long long shard_bytes     = (long long) ((size_t) pages_per_shard * params_.page_bytes);

    // decide whether we can resume an existing folder
    bool resume = false;
    kv_manifest m;
    if (read_manifest(m) && manifest_compatible(m, params_, n_shards_) &&
        m.valid_bitmap.size() == bitmap_bytes) {
        resume = true;
        page_valid_ = m.valid_bitmap;
    }

    shards_.resize(n_shards_);
    for (uint32_t s = 0; s < n_shards_; ++s) {
        const std::string sp = shard_path(s);
        // resume: open existing without truncating; fresh: truncate to empty
        std::FILE * fp = std::fopen(sp.c_str(), resume ? "r+b" : "w+b");
        if (!fp && resume) {
            // a shard went missing: fall back to a fresh folder
            resume = false;
            std::fill(page_valid_.begin(), page_valid_.end(), 0);
            fp = std::fopen(sp.c_str(), "w+b");
        }
        if (!fp) {
            throw std::runtime_error("llama_kv_pager: failed to open shard: " + sp);
        }
        shards_[s].reset(fp);
        if (kv_ftruncate(fp, shard_bytes) != 0) {
            throw std::runtime_error("llama_kv_pager: failed to size shard: " + sp);
        }
    }

    if (resume) {
        // valid pages already live on disk; seed the page table accordingly
        for (uint32_t p = 0; p < params_.n_pages; ++p) {
            if (page_is_valid(p)) {
                pages_[p].tier = LLAMA_KV_TIER_DISK;
            }
        }
    } else {
        write_manifest();
    }
}
```

Also stub `read_manifest`, `write_manifest`, and `manifest_compatible` so the file links — add after the helpers above (these get their real bodies in Tasks 3 and 4; minimal versions here keep the round-trip test honest):

```cpp
bool llama_kv_pager::read_manifest(kv_manifest & /*out*/) const {
    return false; // resume support added in Task 3
}

void llama_kv_pager::write_manifest() {
    // full manifest persistence added in Task 3
}

bool llama_kv_pager::manifest_compatible(const kv_manifest & /*on_disk*/,
                                         const llama_kv_pager_params & /*want*/,
                                         uint32_t /*want_shards*/) {
    return false; // real predicate implemented in Task 4
}
```

- [ ] **Step 7: Make the disk I/O paths shard-aware**

In `src/llama-kv-pager.cpp`, replace `disk_read_page` (lines 109-123) with:

```cpp
void llama_kv_pager::disk_read_page(uint32_t page, void * dst) {
    switch (disk_mode_) {
        case DISK_NONE:
            // a page that is not resident has never been written => logically zero
            memset(dst, 0, params_.page_bytes);
            return;
        case DISK_FILE: {
            if (llama_kv_pager_fseek(file_.get(), (size_t) page * params_.page_bytes, SEEK_SET) != 0) {
                throw std::runtime_error("llama_kv_pager: disk seek failed on read");
            }
            if (std::fread(dst, 1, params_.page_bytes, file_.get()) != params_.page_bytes) {
                throw std::runtime_error("llama_kv_pager: disk read failed");
            }
            stats_.disk_reads++;
            return;
        }
        case DISK_FOLDER: {
            if (!page_is_valid(page)) {
                memset(dst, 0, params_.page_bytes); // never written => zero, no I/O
                return;
            }
            std::FILE * fp = shards_[shard_of(page)].get();
            if (llama_kv_pager_fseek(fp, shard_offset(page), SEEK_SET) != 0) {
                throw std::runtime_error("llama_kv_pager: shard seek failed on read");
            }
            if (std::fread(dst, 1, params_.page_bytes, fp) != params_.page_bytes) {
                throw std::runtime_error("llama_kv_pager: shard read failed");
            }
            stats_.disk_reads++;
            return;
        }
    }
}
```

Replace `disk_write_page` (lines 125-140) with:

```cpp
void llama_kv_pager::disk_write_page(uint32_t page, const void * src) {
    switch (disk_mode_) {
        case DISK_NONE:
            // dropping a clean copy is fine: resident tiers hold every page
            return;
        case DISK_FILE: {
            if (llama_kv_pager_fseek(file_.get(), (size_t) page * params_.page_bytes, SEEK_SET) != 0) {
                throw std::runtime_error("llama_kv_pager: disk seek failed on write");
            }
            if (std::fwrite(src, 1, params_.page_bytes, file_.get()) != params_.page_bytes) {
                throw std::runtime_error("llama_kv_pager: disk write failed");
            }
            std::fflush(file_.get());
            stats_.disk_writes++;
            return;
        }
        case DISK_FOLDER: {
            std::FILE * fp = shards_[shard_of(page)].get();
            if (llama_kv_pager_fseek(fp, shard_offset(page), SEEK_SET) != 0) {
                throw std::runtime_error("llama_kv_pager: shard seek failed on write");
            }
            if (std::fwrite(src, 1, params_.page_bytes, fp) != params_.page_bytes) {
                throw std::runtime_error("llama_kv_pager: shard write failed");
            }
            std::fflush(fp);
            set_page_valid(page);
            stats_.disk_writes++;
            return;
        }
    }
}
```

- [ ] **Step 8: Fix the eviction tier transitions for folder mode**

The two eviction paths currently decide the post-evict tier with `file_ ? DISK : NONE`, which is wrong in folder mode (`file_` is null but a cold tier exists). In `src/llama-kv-pager.cpp`:

In `evict_one_ram` change (was line 166):
```cpp
    m.tier = file_ ? LLAMA_KV_TIER_DISK : LLAMA_KV_TIER_NONE;
```
to:
```cpp
    m.tier = (disk_mode_ != DISK_NONE) ? LLAMA_KV_TIER_DISK : LLAMA_KV_TIER_NONE;
```

In `evict_one_gpu` change (was line 200) the identical line the same way:
```cpp
    m.tier = (disk_mode_ != DISK_NONE) ? LLAMA_KV_TIER_DISK : LLAMA_KV_TIER_NONE;
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed`

- [ ] **Step 10: Commit**

```bash
git add src/llama-kv-pager.h src/llama-kv-pager.cpp tests/test-kv-pager.cpp
git commit -m "kv-pager: folder (sharded) cold tier — construction + striped I/O

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Unwritten folder page reads zero (no I/O)

**Files:**
- Test: `tests/test-kv-pager.cpp`

This locks the contract that a never-written page in a fresh folder reads as zero without a disk read (the valid-bitmap fast path already added in Task 1, Step 7).

- [ ] **Step 1: Add the test**

In `tests/test-kv-pager.cpp`, add before `main`:

```cpp
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
```

Register in `main()` after `test_folder_round_trip();`:

```cpp
    rc |= test_folder_unwritten_reads_zero();
```

- [ ] **Step 2: Run the test**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test-kv-pager.cpp
git commit -m "kv-pager: test unwritten folder page reads zero with no shard I/O

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Resume — manifest persistence + page-table seeding

**Files:**
- Modify: `src/llama-kv-pager.cpp`
- Test: `tests/test-kv-pager.cpp`

Note: this task wires up manifest **persistence** and the **resume** path, but uses a temporary always-true compatibility check so the resume test can pass; the real `manifest_compatible` predicate (including rejection) is implemented in Task 4.

- [ ] **Step 1: Add the resume round-trip test**

In `tests/test-kv-pager.cpp`, add before `main`:

```cpp
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
```

Register in `main()` after `test_folder_unwritten_reads_zero();`:

```cpp
    rc |= test_folder_resume_round_trip();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: **FAIL** at the `tier_of(pg) == LLAMA_KV_TIER_DISK` check — `read_manifest` is still the stub returning `false`, so the rebuild treats the folder as fresh.

- [ ] **Step 3: Implement manifest read/write**

In `src/llama-kv-pager.cpp`, replace the stub `read_manifest` and `write_manifest` (from Task 1, Step 6) with real implementations:

```cpp
bool llama_kv_pager::read_manifest(kv_manifest & out) const {
    const std::string mpath = folder_path_ + "/kv-cache.manifest";
    std::FILE * f = std::fopen(mpath.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::unique_ptr<std::FILE, file_deleter> guard(f);

    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, KV_MANIFEST_MAGIC, 8) != 0) {
        return false;
    }
    uint32_t fp_len = 0;
    if (std::fread(&out.version,    sizeof out.version,    1, f) != 1) return false;
    if (std::fread(&out.page_bytes, sizeof out.page_bytes, 1, f) != 1) return false;
    if (std::fread(&out.n_pages,    sizeof out.n_pages,    1, f) != 1) return false;
    if (std::fread(&out.n_shards,   sizeof out.n_shards,   1, f) != 1) return false;
    if (std::fread(&fp_len,         sizeof fp_len,         1, f) != 1) return false;

    out.fingerprint.resize(fp_len);
    if (fp_len && std::fread(&out.fingerprint[0], 1, fp_len, f) != fp_len) {
        return false;
    }
    const size_t bitmap_bytes = (out.n_pages + 7) / 8;
    out.valid_bitmap.resize(bitmap_bytes);
    if (bitmap_bytes && std::fread(out.valid_bitmap.data(), 1, bitmap_bytes, f) != bitmap_bytes) {
        return false;
    }
    return true;
}

void llama_kv_pager::write_manifest() {
    const std::string mpath = folder_path_ + "/kv-cache.manifest";
    std::FILE * f = std::fopen(mpath.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("llama_kv_pager: failed to open manifest for write: " + mpath);
    }
    std::unique_ptr<std::FILE, file_deleter> guard(f);

    const uint32_t version = KV_MANIFEST_VERSION;
    const uint64_t pb      = params_.page_bytes;
    const uint32_t fp_len  = (uint32_t) params_.disk_fingerprint.size();

    bool ok = true;
    ok = ok && std::fwrite(KV_MANIFEST_MAGIC, 1, 8, f) == 8;
    ok = ok && std::fwrite(&version,         sizeof version,         1, f) == 1;
    ok = ok && std::fwrite(&pb,              sizeof pb,              1, f) == 1;
    ok = ok && std::fwrite(&params_.n_pages, sizeof params_.n_pages, 1, f) == 1;
    ok = ok && std::fwrite(&n_shards_,       sizeof n_shards_,       1, f) == 1;
    ok = ok && std::fwrite(&fp_len,          sizeof fp_len,          1, f) == 1;
    if (fp_len) {
        ok = ok && std::fwrite(params_.disk_fingerprint.data(), 1, fp_len, f) == fp_len;
    }
    if (!page_valid_.empty()) {
        ok = ok && std::fwrite(page_valid_.data(), 1, page_valid_.size(), f) == page_valid_.size();
    }
    std::fflush(f);
    if (!ok) {
        throw std::runtime_error("llama_kv_pager: failed to write manifest body: " + mpath);
    }
}
```

- [ ] **Step 4: Persist the manifest on flush()**

In `src/llama-kv-pager.cpp`, at the end of `flush()` (after the `for` loop, before the closing brace ~line 342), add:

```cpp
    if (disk_mode_ == DISK_FOLDER) {
        write_manifest(); // persist the valid bitmap so a restart can resume
    }
```

- [ ] **Step 5: Temporarily allow resume so the test can pass**

In `src/llama-kv-pager.cpp`, change the Task 1 stub `manifest_compatible` to accept matching geometry (the full predicate, including fingerprint handling and rejection, lands in Task 4):

```cpp
bool llama_kv_pager::manifest_compatible(const kv_manifest & on_disk,
                                         const llama_kv_pager_params & want,
                                         uint32_t want_shards) {
    // TEMPORARY (Task 4 replaces this): accept only exact geometry matches.
    return on_disk.version    == KV_MANIFEST_VERSION
        && on_disk.page_bytes == want.page_bytes
        && on_disk.n_pages    == want.n_pages
        && on_disk.n_shards   == want_shards;
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed`

- [ ] **Step 7: Commit**

```bash
git add src/llama-kv-pager.cpp tests/test-kv-pager.cpp
git commit -m "kv-pager: persist folder manifest + resume page table from disk

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Resume-safety predicate (`manifest_compatible`) — LEARNING-MODE HANDOFF

**Files:**
- Modify: `src/llama-kv-pager.cpp`
- Test: `tests/test-kv-pager.cpp`

> **Learning-mode handoff:** the resume-safety predicate is a judgment call (which mismatches are fatal vs. recoverable, how an empty fingerprint is treated). During execution, present the scaffold below to the user and let them write the body (~5-8 lines). The reference implementation in Step 3 is the oracle the test must satisfy — use it to validate the user's version, or as the fallback if they decline.

- [ ] **Step 1: Add the resume-rejection test**

In `tests/test-kv-pager.cpp`, add before `main`:

```cpp
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
    };

    // 1) different fingerprint -> fresh
    seed_folder("model-A");
    reopen_expects_fresh("model-B", 3);

    // 2) different shard count (geometry) -> fresh
    seed_folder("model-A");
    reopen_expects_fresh("model-A", 4);

    // 3) empty fingerprint -> resume not requested -> fresh
    seed_folder("model-A");
    reopen_expects_fresh("", 3);

    remove_dir(dir);
    return 0;
}
```

Register in `main()` after `test_folder_resume_round_trip();`:

```cpp
    rc |= test_folder_resume_rejected_on_mismatch();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: **FAIL** — the temporary predicate from Task 3 ignores the fingerprint, so the "different fingerprint" and "empty fingerprint" cases wrongly resume and the probe page is seeded as `DISK`.

- [ ] **Step 3: Implement the predicate (user-written; reference below)**

Replace the temporary `manifest_compatible` in `src/llama-kv-pager.cpp` with the scaffold, and fill in the `TODO`:

```cpp
// Decide whether an existing on-disk manifest may be reused for the current run.
// Returns true only if reusing the folder's shards is safe; false forces a fresh
// (scratch) folder. Geometry mismatches (page_bytes / n_pages / n_shards) must be
// fatal-to-reuse; a supplied fingerprint must match; an empty current fingerprint
// means resume was not requested (return false).
bool llama_kv_pager::manifest_compatible(const kv_manifest & on_disk,
                                         const llama_kv_pager_params & want,
                                         uint32_t want_shards) {
    // TODO(user): implement the compatibility check (~5-8 lines).
}
```

Reference implementation (oracle / fallback):

```cpp
bool llama_kv_pager::manifest_compatible(const kv_manifest & on_disk,
                                         const llama_kv_pager_params & want,
                                         uint32_t want_shards) {
    if (want.disk_fingerprint.empty())                   return false; // resume not requested
    if (on_disk.version     != KV_MANIFEST_VERSION)      return false;
    if (on_disk.page_bytes  != want.page_bytes)          return false;
    if (on_disk.n_pages     != want.n_pages)             return false;
    if (on_disk.n_shards    != want_shards)              return false;
    if (on_disk.fingerprint != want.disk_fingerprint)    return false;
    return true;
}
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed` (resume round-trip from Task 3 still passes — its fingerprint matches.)

- [ ] **Step 5: Commit**

```bash
git add src/llama-kv-pager.cpp tests/test-kv-pager.cpp
git commit -m "kv-pager: resume-safety predicate + rejection tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Invalid / edge configuration handling

**Files:**
- Modify: `src/llama-kv-pager.cpp`
- Test: `tests/test-kv-pager.cpp`

- [ ] **Step 1: Add the edge-config tests**

In `tests/test-kv-pager.cpp`, add before `main`:

```cpp
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
```

Register in `main()` after `test_folder_resume_rejected_on_mismatch();`:

```cpp
    rc |= test_folder_invalid_config();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: **FAIL** — `open_or_create_folder` does not yet reject an empty path (the first sub-case constructs without throwing; with an empty `folder_path_`, `kv_exists("")` is false and `kv_mkdir("")` may or may not error, so the behavior is unreliable).

- [ ] **Step 3: Reject an empty folder path explicitly**

In `src/llama-kv-pager.cpp`, at the very top of `open_or_create_folder()` (before resolving `folder_path_`), add:

```cpp
    if (params_.disk_path.empty()) {
        throw std::runtime_error("llama_kv_pager: disk_shards >= 1 requires a directory disk_path");
    }
```

(The not-a-directory case is already handled by the existing `kv_isdir` check, and clamping is already done by the `n_shards_ > params_.n_pages` guard — this test locks both in.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --target test-kv-pager -j && ./build/bin/test-kv-pager`
Expected: `test-kv-pager: all tests passed`

- [ ] **Step 5: Commit**

```bash
git add src/llama-kv-pager.cpp tests/test-kv-pager.cpp
git commit -m "kv-pager: validate folder config (empty path, non-dir, shard clamp)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Config surface — `kv_disk_shards`

**Files:**
- Modify: `include/llama.h`
- Modify: `src/llama-memory.h`
- Modify: `src/llama-context.cpp`
- Modify: `src/llama-model.cpp`
- Modify: `src/llama-kv-cache.h`
- Modify: `src/llama-kv-cache.cpp`
- Modify: `common/common.h`
- Modify: `common/common.cpp`
- Modify: `common/arg.cpp`

This plumbs a new option end-to-end. It has **no runtime effect yet** (the pager cold tier is not in the live decode path) — it is config-ready surface, mirroring how `kv_offload_disk`/`kv_disk_path` were introduced.

- [ ] **Step 1: `include/llama.h` — add the field**

After the `kv_prefetch_depth` line in `llama_context_params` (line 400):

```cpp
        uint32_t     kv_disk_shards;    // shard files for a folder cold tier (0 => single file; folder path => default 8)
```

- [ ] **Step 2: `src/llama-memory.h` — add the field**

After the `kv_prefetch_depth` line in `llama_memory_params` (line 34):

```cpp
    uint32_t    kv_disk_shards;   // shard files for a folder cold tier (0 => single file)
```

- [ ] **Step 3: `src/llama-context.cpp` — pass through + default**

In the `params_mem` initializer, after the `/*.kv_prefetch_depth =*/ params.kv_prefetch_depth,` line (line 315):

```cpp
            /*.kv_disk_shards    =*/ params.kv_disk_shards,
```

In the default-params block, after `/*.kv_prefetch_depth           =*/ 0,` (line 3416):

```cpp
        /*.kv_disk_shards              =*/ 0,
```

- [ ] **Step 4: `src/llama-kv-cache.h` — accept + store (inert)**

In the constructor declaration, change the `kv_disk_path` parameter line (line 111) to add a trailing param:

```cpp
              const std::string & kv_disk_path = std::string(),
                         uint32_t kv_disk_shards = 0);
```

After the `const std::string kv_disk_path;` member (line 264):

```cpp
    const uint32_t    kv_disk_shards = 0; // folder cold-tier shard count (config-ready; not yet live)
```

- [ ] **Step 5: `src/llama-kv-cache.cpp` — constructor**

Change the constructor signature's `kv_disk_path` parameter line (line 102) to:

```cpp
      const std::string &   kv_disk_path,
                 uint32_t   kv_disk_shards) :
```

Add to the member-initializer list — change the tail (line 105) from:

```cpp
    swa_type(swa_type), kv_offload_disk(kv_offload_disk), kv_disk_path(kv_disk_path) {
```

to:

```cpp
    swa_type(swa_type), kv_offload_disk(kv_offload_disk), kv_disk_path(kv_disk_path),
    kv_disk_shards(kv_disk_shards) {
```

- [ ] **Step 6: `src/llama-model.cpp` — pass the value at the call site**

Change the kv-cache construction tail (line 2099) from:

```cpp
                                params.kv_disk_path ? params.kv_disk_path : "");
```

to:

```cpp
                                params.kv_disk_path ? params.kv_disk_path : "",
                                params.kv_disk_shards);
```

- [ ] **Step 7: `common/common.h` — add the field**

After the `kv_disk_path` field (line 549):

```cpp
    int32_t     kv_disk_shards  = 0;     // folder cold-tier shard count (0 => single file; folder path => default 8)
```

- [ ] **Step 8: `common/common.cpp` — set cparams + folder default**

After the `cparams.kv_page_tokens = (uint32_t) params.kv_page_tokens;` line (line 1597), add:

```cpp
    cparams.kv_disk_shards    = (uint32_t) params.kv_disk_shards;
    // a folder cold tier with an unset shard count gets a sensible default
    if (params.kv_offload_disk && cparams.kv_disk_shards == 0 && !params.kv_disk_path.empty()) {
        struct stat st = {};
        if (stat(params.kv_disk_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            cparams.kv_disk_shards = 8;
        }
    }
```

(`<sys/stat.h>` is already included in `common/common.cpp` at line 49.)

- [ ] **Step 9: `common/arg.cpp` — add `--kv-disk-shards`**

After the `--kv-disk-path` option block (the `add_opt(...).set_env("LLAMA_ARG_KV_DISK_PATH"));` ending at line 2040), add:

```cpp
    add_opt(common_arg(
        {"--kv-disk-shards"}, "N",
        "number of shard files for a folder-based disk-backed KV cold tier; implies a directory --kv-disk-path (default: 8 when a folder is given)",
        [](common_params & params, int value) {
            params.kv_disk_shards = value;
            params.kv_offload_disk = true;
        }
    ).set_env("LLAMA_ARG_KV_DISK_SHARDS"));
```

- [ ] **Step 10: Build the library + CLI and verify the flag is wired**

Run: `cmake --build build --target llama-cli -j`
Expected: build succeeds (no errors).

Run: `./build/bin/llama-cli --help | grep -A2 "kv-disk-shards"`
Expected: the `--kv-disk-shards N` option and its description are printed.

- [ ] **Step 11: Commit**

```bash
git add include/llama.h src/llama-memory.h src/llama-context.cpp src/llama-model.cpp \
        src/llama-kv-cache.h src/llama-kv-cache.cpp common/common.h common/common.cpp common/arg.cpp
git commit -m "config: plumb kv_disk_shards through to the KV cache (config-ready)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Documentation

**Files:**
- Modify: `docs/kv-cache-disk-offload.md`

- [ ] **Step 1: Add `kv_disk_shards` to the configuration table**

In `docs/kv-cache-disk-offload.md`, in the Configuration table (the rows around line 140-147), add a row after the `kv_prefetch_depth` row:

```markdown
| `kv_disk_shards`    | shard files for a folder cold tier (0 ⇒ single file; folder path ⇒ default 8) |
```

- [ ] **Step 2: Add a short folder/resume note**

In the same file, immediately after the configuration table (after line 147), add:

```markdown
### Folder (sharded) cold tier

When `kv_disk_path` names a directory and `kv_disk_shards >= 1`, the pager's cold
tier stripes pages across `kv_disk_shards` shard files inside that folder
(`shard = page % n_shards`, `offset = (page / n_shards) * page_bytes`) instead of
using a single pre-sized file. The shards are created sparse (no startup
pre-zeroing). A `kv-cache.manifest` in the folder records the geometry, a
caller-supplied fingerprint, and a per-page valid bitmap, so a folder can be
**resumed** across runs: on a matching fingerprint + geometry the written pages
are restored to the disk tier; on any mismatch the folder is treated as fresh.
This is implemented in the pager engine (`src/llama-kv-pager.{h,cpp}`) and is
exercised by `tests/test-kv-pager.cpp`; like the rest of the cold tier it is not
yet wired into the live decode path.
```

- [ ] **Step 3: Commit**

```bash
git add docs/kv-cache-disk-offload.md
git commit -m "docs: document the folder (sharded) resumable KV cold tier

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the pager test suite via CTest**

Run: `ctest --test-dir build -R test-kv-pager --output-on-failure`
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 2: Confirm the new tests are all present**

Run: `./build/bin/test-kv-pager && echo OK`
Expected: `test-kv-pager: all tests passed` then `OK`. The run covers: single-file round-trip (unchanged), folder round-trip, unwritten-reads-zero, resume round-trip, resume rejection, and invalid/edge configs.

- [ ] **Step 3: Spec coverage self-check**

Confirm each spec section maps to a task: §1 params/mode → Task 1; §2 sharded layout/sparse → Task 1; §3 resume/manifest → Tasks 3-4; §4 config surface → Task 6; §5 tests → Tasks 1-5; §6 `manifest_compatible` handoff → Task 4. No gaps.
