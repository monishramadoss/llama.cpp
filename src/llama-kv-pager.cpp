#include "llama-kv-pager.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>

// portable 64-bit file seek (large page-backing files can exceed 2 GiB)
#if defined(_WIN32)
#   define llama_kv_pager_fseek(stream, offset, whence) _fseeki64(stream, (__int64) (offset), whence)
#elif defined(__ANDROID__) || defined(__APPLE__) || defined(__linux__) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
#   define llama_kv_pager_fseek(stream, offset, whence) fseeko(stream, (off_t) (offset), whence)
#else
#   define llama_kv_pager_fseek(stream, offset, whence) std::fseek(stream, (long) (offset), whence)
#endif

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

llama_kv_pager_io llama_kv_pager_io::make_default() {
    llama_kv_pager_io io;
    io.copy_ram_to_gpu = [](void * dst, const void * src, size_t n) { memcpy(dst, src, n); };
    io.copy_gpu_to_ram = [](void * dst, const void * src, size_t n) { memcpy(dst, src, n); };
    return io;
}

void llama_kv_pager::pool::init(uint32_t cap, size_t pb) {
    capacity   = cap;
    page_bytes = pb;
    data.assign((size_t) cap * pb, 0);
    slot_page.assign(cap, -1);
}

void * llama_kv_pager::pool::slot_ptr(int32_t slot) {
    assert(slot >= 0 && (uint32_t) slot < capacity);
    return data.data() + (size_t) slot * page_bytes;
}

int32_t llama_kv_pager::pool::find_free() const {
    for (uint32_t i = 0; i < capacity; ++i) {
        if (slot_page[i] < 0) {
            return (int32_t) i;
        }
    }
    return -1;
}

llama_kv_pager::llama_kv_pager(const llama_kv_pager_params & params, llama_kv_pager_io io)
    : params_(params), io_(std::move(io)) {
    if (params_.page_bytes == 0) {
        throw std::runtime_error("llama_kv_pager: page_bytes must be > 0");
    }
    if (params_.n_pages == 0) {
        throw std::runtime_error("llama_kv_pager: n_pages must be > 0");
    }
    if (params_.gpu_pages == 0) {
        throw std::runtime_error("llama_kv_pager: gpu_pages must be > 0");
    }
    if (!io_.copy_ram_to_gpu || !io_.copy_gpu_to_ram) {
        throw std::runtime_error("llama_kv_pager: copy hooks must be set");
    }

    gpu_.init(params_.gpu_pages, params_.page_bytes);
    ram_.init(params_.ram_pages, params_.page_bytes);

    pages_.assign(params_.n_pages, page_meta{});

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
        // are valid up front (avoids fragmentation and out-of-range seeks).
        // NOTE: "w+b" truncates any existing file, which is the desired
        // behavior for a fresh scratch tier.
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

    if (params_.verbose) {
        fprintf(stderr,
            "%s: page_bytes=%zu n_pages=%u gpu_pages=%u ram_pages=%u disk=%s prefetch=%u\n",
            __func__, params_.page_bytes, params_.n_pages, params_.gpu_pages, params_.ram_pages,
            params_.disk_path.empty() ? "(none)" : params_.disk_path.c_str(), params_.prefetch_depth);
    }
}

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

// Decide whether an existing on-disk manifest may be reused for the current run.
// Returns true only if reusing the folder's shards is safe; false forces a fresh
// (scratch) folder. Geometry mismatches (page_bytes / n_pages / n_shards) are
// fatal-to-reuse; a supplied fingerprint must match; an empty current fingerprint
// means resume was not requested.
//
// NOTE: this is the learning-mode handoff point from the plan (Task 4). The body
// below is the reference predicate; swap in your own policy if you want different
// resume semantics (e.g. tolerate a version bump, or ignore the fingerprint).
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

void llama_kv_pager::open_or_create_folder() {
    if (params_.disk_path.empty()) {
        throw std::runtime_error("llama_kv_pager: disk_shards >= 1 requires a directory disk_path");
    }

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

llama_kv_pager::~llama_kv_pager() = default;

llama_kv_tier llama_kv_pager::tier_of(uint32_t page) const {
    assert(page < pages_.size());
    return pages_[page].tier;
}

bool llama_kv_pager::is_fully_resident() const {
    return (uint64_t) params_.gpu_pages + params_.ram_pages >= params_.n_pages;
}

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
            // flush so that a subsequent fread (after an fseek) observes the new bytes
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

int32_t llama_kv_pager::lru_page_in_tier(llama_kv_tier tier) const {
    int32_t  victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (size_t p = 0; p < pages_.size(); ++p) {
        if (pages_[p].tier == tier && pages_[p].last_use < oldest) {
            oldest = pages_[p].last_use;
            victim = (int32_t) p;
        }
    }
    return victim;
}

int32_t llama_kv_pager::evict_one_ram() {
    const int32_t victim = lru_page_in_tier(LLAMA_KV_TIER_RAM);
    if (victim < 0) {
        throw std::runtime_error("llama_kv_pager: no RAM page to evict");
    }
    page_meta & m = pages_[victim];
    if (m.dirty) {
        disk_write_page((uint32_t) victim, ram_.slot_ptr(m.slot));
        m.dirty = false;
    }
    const int32_t slot = m.slot;
    ram_.slot_page[slot] = -1;
    m.tier = (disk_mode_ != DISK_NONE) ? LLAMA_KV_TIER_DISK : LLAMA_KV_TIER_NONE;
    m.slot = -1;
    stats_.ram_evictions++;
    return slot;
}

int32_t llama_kv_pager::evict_one_gpu() {
    const int32_t victim = lru_page_in_tier(LLAMA_KV_TIER_GPU);
    if (victim < 0) {
        throw std::runtime_error("llama_kv_pager: no GPU page to evict");
    }
    page_meta & m = pages_[victim];
    const int32_t gpu_slot = m.slot;

    if (ram_.capacity > 0) {
        // spill down to RAM (staging tier)
        int32_t ram_slot = ram_.find_free();
        if (ram_slot < 0) {
            ram_slot = evict_one_ram();
        }
        io_.copy_gpu_to_ram(ram_.slot_ptr(ram_slot), gpu_.slot_ptr(gpu_slot), params_.page_bytes);
        ram_.slot_page[ram_slot] = victim;
        m.tier = LLAMA_KV_TIER_RAM;
        m.slot = ram_slot;
        // dirty flag carries over: the RAM copy is now the freshest one
    } else {
        // no warm tier: spill straight to disk
        if (m.dirty) {
            // pull the freshest bytes back to a temporary then write to disk
            std::vector<uint8_t> tmp(params_.page_bytes);
            io_.copy_gpu_to_ram(tmp.data(), gpu_.slot_ptr(gpu_slot), params_.page_bytes);
            disk_write_page((uint32_t) victim, tmp.data());
            m.dirty = false;
        }
        m.tier = (disk_mode_ != DISK_NONE) ? LLAMA_KV_TIER_DISK : LLAMA_KV_TIER_NONE;
        m.slot = -1;
    }

    gpu_.slot_page[gpu_slot] = -1;
    stats_.gpu_evictions++;
    return gpu_slot;
}

int32_t llama_kv_pager::ensure_ram(uint32_t page) {
    page_meta & m = pages_[page];
    if (m.tier == LLAMA_KV_TIER_RAM) {
        return m.slot;
    }
    if (ram_.capacity == 0) {
        throw std::runtime_error("llama_kv_pager: ensure_ram called without a RAM tier");
    }

    int32_t ram_slot = ram_.find_free();
    if (ram_slot < 0) {
        ram_slot = evict_one_ram();
    }

    // bring data into the RAM slot from its current tier
    if (m.tier == LLAMA_KV_TIER_GPU) {
        io_.copy_gpu_to_ram(ram_.slot_ptr(ram_slot), gpu_.slot_ptr(m.slot), params_.page_bytes);
        gpu_.slot_page[m.slot] = -1;
        // dirty carries over to RAM
    } else { // DISK or NONE
        disk_read_page(page, ram_.slot_ptr(ram_slot));
    }

    ram_.slot_page[ram_slot] = (int32_t) page;
    m.tier = LLAMA_KV_TIER_RAM;
    m.slot = ram_slot;
    return ram_slot;
}

void * llama_kv_pager::acquire_gpu(uint32_t page) {
    if (page >= pages_.size()) {
        throw std::runtime_error("llama_kv_pager: acquire_gpu page out of range");
    }
    page_meta & m = pages_[page];

    if (m.tier == LLAMA_KV_TIER_GPU) {
        stats_.gpu_hits++;
        m.last_use = ++clock_;
        return gpu_.slot_ptr(m.slot);
    }

    int32_t gpu_slot = gpu_.find_free();
    if (gpu_slot < 0) {
        gpu_slot = evict_one_gpu();
    }

    // stage the page up into the GPU slot
    if (m.tier == LLAMA_KV_TIER_RAM) {
        stats_.ram_hits++;
        io_.copy_ram_to_gpu(gpu_.slot_ptr(gpu_slot), ram_.slot_ptr(m.slot), params_.page_bytes);
        ram_.slot_page[m.slot] = -1;
        // dirty carries over to the GPU copy
    } else if (ram_.capacity > 0) {
        // go through the warm tier (disk -> RAM -> GPU)
        const int32_t ram_slot = ensure_ram(page);
        io_.copy_ram_to_gpu(gpu_.slot_ptr(gpu_slot), ram_.slot_ptr(ram_slot), params_.page_bytes);
        ram_.slot_page[ram_slot] = -1;
    } else {
        // no warm tier: read straight from disk into a temp then to GPU
        std::vector<uint8_t> tmp(params_.page_bytes);
        disk_read_page(page, tmp.data());
        io_.copy_ram_to_gpu(gpu_.slot_ptr(gpu_slot), tmp.data(), params_.page_bytes);
    }

    gpu_.slot_page[gpu_slot] = (int32_t) page;
    m.tier = LLAMA_KV_TIER_GPU;
    m.slot = gpu_slot;
    m.last_use = ++clock_;

    if (params_.prefetch_depth > 0) {
        for (uint32_t d = 1; d <= params_.prefetch_depth; ++d) {
            const uint32_t np = page + d;
            if (np < pages_.size()) {
                prefetch(np, /*to_gpu =*/ false);
            }
        }
    }

    return gpu_.slot_ptr(gpu_slot);
}

void llama_kv_pager::mark_dirty(uint32_t page) {
    if (page >= pages_.size()) {
        throw std::runtime_error("llama_kv_pager: mark_dirty page out of range");
    }
    page_meta & m = pages_[page];
    if (m.tier != LLAMA_KV_TIER_GPU) {
        throw std::runtime_error("llama_kv_pager: mark_dirty on a page that is not GPU-resident");
    }
    m.dirty = true;
}

void llama_kv_pager::prefetch(uint32_t page, bool to_gpu) {
    if (page >= pages_.size()) {
        return;
    }
    page_meta & m = pages_[page];
    if (m.tier == LLAMA_KV_TIER_GPU) {
        return;
    }
    if (to_gpu) {
        acquire_gpu(page);
        return;
    }
    if (ram_.capacity > 0 && m.tier != LLAMA_KV_TIER_RAM) {
        // only stage into RAM if there is a free slot; never evict on prefetch
        // to avoid thrashing pages that are actively in use.
        if (ram_.find_free() >= 0) {
            ensure_ram(page);
        }
    }
}

void llama_kv_pager::flush() {
    for (uint32_t p = 0; p < pages_.size(); ++p) {
        page_meta & m = pages_[p];
        if (!m.dirty) {
            continue;
        }
        const void * src = nullptr;
        std::vector<uint8_t> tmp;
        if (m.tier == LLAMA_KV_TIER_GPU) {
            tmp.resize(params_.page_bytes);
            io_.copy_gpu_to_ram(tmp.data(), gpu_.slot_ptr(m.slot), params_.page_bytes);
            src = tmp.data();
        } else if (m.tier == LLAMA_KV_TIER_RAM) {
            src = ram_.slot_ptr(m.slot);
        }
        if (src) {
            disk_write_page(p, src);
            m.dirty = false;
        }
    }

    if (disk_mode_ == DISK_FOLDER) {
        write_manifest(); // persist the valid bitmap so a restart can resume
    }
}
