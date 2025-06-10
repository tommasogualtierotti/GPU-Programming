# GPU Hashing Benchmark

*Comparative performance analysis of MD5, SHA‑1 & SHA‑256 on CPU **vs** CUDA GPU*

---

> **TL;DR —** A mid‑range **NVIDIA GTX 1650** running three overlapping CUDA streams hashes **18 million strings in under one second**, roughly **10 × faster** than an 8‑core Ryzen 7 3700X.

![CUDA 11 +](https://img.shields.io/badge/CUDA-11%2B-green) ![License: MIT](https://img.shields.io/badge/License-MIT-blue)

---

## Overview

This repository offers —

* **Reference C implementations** of MD5, SHA‑1 and SHA‑256 for verification and CPU benchmarking.
* **Optimised CUDA kernels** that execute the same hashes massively in parallel.
* **Utility helpers** for batched I/O, timing, and result checking.
* **A Make‑based build system** targeting both CPU‑only and GPU‑accelerated binaries.
* **A full report (PDF)** capturing methodology, tuning sweeps, roofline plots and raw data.

The guiding question is **“How much throughput can a commodity GPU deliver for bulk hashing, and which optimisation levers matter most?”**

---

## Features

| Feature                          | Description                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Batch processing**             | Streams arbitrarily large datasets through a small, reusable buffer ‑ runs even on 4 GB cards.  |
| **Triple‑buffer CUDA streams**   | Overlaps Host→Device copies, kernel execution and Device→Host copies for near­‑max utilisation. |
| **One‑line configuration**       | Flip batching, streams, block size & more in `include/config.h` — no source edits needed.       |
| **Cross‑checked correctness**    | CPU and GPU digests are compared at runtime; any mismatch aborts execution.                     |
| **Lightweight datasets via LFS** | Large test files reside in Git LFS so the repository clone stays trim.                          |

---

## Project Structure

```
.
├── Makefile              # Targets: cpu, gpu, gpu_streams, clean …
├── include/              # Public headers & switches
│   └── config.h          # ✏️  Dataset path, batch size, stream count…
├── src/
│   ├── cpu source files  # md5.c, sha1.c, sha256.c  (reference)
│   ├── gpu source files  # md5_kernel.cu, …         (CUDA kernels)
│   └── utilities.c       # I/O helpers, timers, pretty printers
├── dataset/              # Large CSV test files (Git LFS)
└── report/
    └── report.pdf        # Full write‑up with graphs & analysis
```

---

## Prerequisites

* **CUDA Toolkit 11.0 +**
* **GPU Compute Capability 7.5 +** (GTX 1650 or newer)
* **GCC 9 +/ Clang 11 +** (tested with GCC 13)
* **Git LFS** for dataset checkout
* *Optional:* **CMake** if you prefer over Make

```bash
# Enable LFS once per machine
git lfs install
```

---

# How to clone the repository

Since dataset files are big, they are stored in the repository using `lfs`, so if the repository is cloned using ssh ensure to own `git lfs` with the command `git lfs --version`. If `git lfs` is not present then it can be installed using `git lfs install`. If `git lfs` cannot be used / downloaded, then it is required to download manually the datasets from the repository as raw files. \
Please note that in case of repository downloading as `zip file`, the dataset still need to be download as raw files from the dataset folder.

# Quick start

1. Extract a file from the dataset folder and put the name of the file in the `include/config.h` file under the `#define FILENAME_PATH` macro.
2. Edit the Makefile according to the features of the GPU used to run the code (more specifically edit the root directory of CUDA binaries and the SM_ARCH flag).
3. Edit the `include/config.h` file, and set the configuration to run the code, i.e. decide whether to use STREAM execution or not, whether to use the batch processing (if the dataset is too large to fit either in the CPU or the GPU ram memory) and so on.
4. Now type `make` in the terminal and then run the binary generated using the command `./exe/test`.


## Configuration (`include/config.h`)

| Macro             | Role                              | Default                   |
| ----------------- | --------------------------------- | ------------------------- |
| `FILENAME_PATH`   | Input file path                   | `"dataset/18M_lines.txt"` |
| `USE_STREAMS`     | Enable CUDA stream overlap        | `1`                       |
| `STREAM_COUNT`    | Concurrent streams                | `3`                       |
| `USE_BATCHING`    | Recycle fixed‑size buffers        | `1`                       |
| `BATCH_NUM_LINES` | Lines per batch (if batching = 1) | `131072`                  |
| `BLOCK_SIZE`      | CUDA threads per block            | `256`                     |

Re‑run `make` after editing.

---

## Benchmarks

*Hardware:* **Ryzen 7 3700X @ 4.4 GHz** vs **GTX 1650 (896 cores, 4 GB, CUDA 12)** using default config.

|  18 M lines | CPU (ms) | GPU (ms) | GPU + Streams (ms) |   Speed‑up |
| ----------: | -------: | -------: | -----------------: | ---------: |
|     **MD5** |    7 804 |    1 107 |            **804** |  **9.7 ×** |
|   **SHA‑1** |    9 467 |    1 203 |            **884** | **10.7 ×** |
| **SHA‑256** |   10 234 |    1 309 |            **899** | **11.4 ×** |

🛈 *For < 1 M lines the CPU wins due to GPU launch overhead; beyond that, the GPU dominates.*

Detailed graphs and profiler screenshots live in [`report/report.pdf`](report/gpu_programming_report_group2.pdf).

---

## Profiling Insights

* Kernels achieve **63‑70 %** of peak integer ALU throughput while using **< 50 %** of memory bandwidth → *compute‑bound*.
* Triple‑buffer streaming offers a **15‑20 %** free boost by hiding PCIe latency.
* A block size of 256 threads strikes the best occupancy/pressure balance.

---

## Limitations

1. Accelerates **large batches only**; single‑hash latency is still CPU‑faster.
2. MD5 & SHA‑1 are **insecure**; keep them for benchmarks, use SHA‑256/SHA‑3 in production.
3. CPU SHA‑extensions were **disabled** in the baseline; enabling them narrows the gap.


## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

## Authors
- [Tommaso Gualtierotti](https://github.com/tommasogualtierotti)
- [Simone Mulazzi](https://github.com/mulaz1)

**Happy hashing!** 🚀
