/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/file_io.hpp>

#include <algorithm>
#include <cstring>
#include <limits.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace cuvs::util {

namespace {
// Writes at or above this size are eligible for page-cache eviction (when the caller opts in via
// write_large_file's drop_from_cache flag). Large intermediate artifacts (dataset, graph,
// reordered/augmented .npy files) would otherwise balloon the page cache and increase
// physical-memory fragmentation, which makes subsequent high-order kernel allocations
// fail (e.g. the page-pointer array the NVIDIA driver kmallocs to longterm-pin a large
// cuMemAllocHost buffer). Small metadata writes are left in cache to avoid extra syncs.
constexpr size_t kPageCacheEvictionThreshold = size_t{32} * 1024 * 1024;
}  // namespace

void drop_file_range_from_cache(const file_descriptor& fd,
                                const uint64_t file_offset,
                                const size_t total_bytes)
{
  if (!fd.is_valid() || total_bytes == 0) { return; }
  // fdatasync first so the dirty pages are flushed to stable storage and therefore eligible for
  // eviction. Best-effort: a failure here does not corrupt the file, so it is non-fatal.
  if (fdatasync(fd.get()) == 0) {
    posix_fadvise(fd.get(),
                  static_cast<off_t>(file_offset),
                  static_cast<off_t>(total_bytes),
                  POSIX_FADV_DONTNEED);
  }
}

void read_large_file(const file_descriptor& fd,
                     void* dest_ptr,
                     const size_t total_bytes,
                     const uint64_t file_offset)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(dest_ptr != nullptr, "Destination pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");

  const size_t read_chunk_size = std::min<size_t>(1024 * 1024 * 1024, SSIZE_MAX);
  size_t bytes_remaining       = total_bytes;
  size_t offset                = 0;

  while (bytes_remaining > 0) {
    const size_t chunk_size = std::min(read_chunk_size, bytes_remaining);
    const uint64_t file_pos = file_offset + offset;
    const ssize_t bytes_read =
      pread(fd.get(), reinterpret_cast<char*>(dest_ptr) + offset, chunk_size, file_pos);

    RAFT_EXPECTS(
      bytes_read != -1, "Failed to read from file at offset %lu: %s", file_pos, strerror(errno));
    RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(chunk_size),
                 "Incomplete read from file. Expected %zu bytes, got %zd at offset %lu",
                 chunk_size,
                 bytes_read,
                 file_pos);

    bytes_remaining -= chunk_size;
    offset += chunk_size;
  }
}

void write_large_file(const file_descriptor& fd,
                      const void* data_ptr,
                      const size_t total_bytes,
                      const uint64_t file_offset,
                      bool drop_from_cache)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(data_ptr != nullptr, "Data pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");

  const size_t write_chunk_size = std::min<size_t>(1024 * 1024 * 1024, SSIZE_MAX);
  size_t bytes_remaining        = total_bytes;
  size_t offset                 = 0;

  while (bytes_remaining > 0) {
    const size_t chunk_size = std::min(write_chunk_size, bytes_remaining);
    const uint64_t file_pos = file_offset + offset;
    const ssize_t chunk_written =
      pwrite(fd.get(), reinterpret_cast<const char*>(data_ptr) + offset, chunk_size, file_pos);

    RAFT_EXPECTS(
      chunk_written != -1, "Failed to write to file at offset %lu: %s", file_pos, strerror(errno));
    RAFT_EXPECTS(chunk_written == static_cast<ssize_t>(chunk_size),
                 "Incomplete write to file. Expected %zu bytes, wrote %zd at offset %lu",
                 chunk_size,
                 chunk_written,
                 file_pos);

    bytes_remaining -= chunk_size;
    offset += chunk_size;
  }

  // Opt-in: for large writes, drop the just-written range from the page cache to keep the cache
  // small and limit physical-memory fragmentation during a build. Off by default so callers that
  // re-read what they just wrote (e.g. spilled temp files) are not penalized with cold reads, and
  // so a per-batch caller does not pay a full-file fdatasync per batch.
  if (drop_from_cache && total_bytes >= kPageCacheEvictionThreshold) {
    drop_file_range_from_cache(fd, file_offset, total_bytes);
  }
}

}  // namespace cuvs::util
