# Lab 2 — SGEMM ladder

Working report draft, November 9, 2026. Predictions below are based on the
source and the A100 specifications in the lab README. Measured comparisons
remain pending. The Part 4 prediction table now includes the LS6 compiler
register count. Preserve these predictions before running `make sweep` or
`make ncu-sweep`, and keep them unchanged afterward.
The final PDF must be six pages or fewer.

## Part 0 — Setup

Existing setup output (`part0/setup_cublas.txt`):

```text
# host c301-003.ls6.tacc.utexas.edu
# device NVIDIA A100-PCIE-40GB  CC 8.0  108 SMs  1410 MHz  39 GiB
```

Compiler output (`part0/nvcc_version.txt`):

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

The counter check in `part0/counter_check.txt` reported
`sm__cycles_elapsed.avg = 178,961.54 cycle` for the cuBLAS GEMM kernel.
Counter access worked. These are setup records; final performance tables
will come from the final batch run.

## Part 1 — Naive

### Prediction before profiling

Let lane `l` range from 0 to 31, with first row `r` and fixed column `c`.
Because x selects the row, the addresses at inner-loop index `k` are:

- A: `&A[(r + l) * N + k]`. The 32 lanes access 32 different sectors.
- B: `&B[k * N + c]`. All lanes access one word in one sector.

Each sector holds eight floats. Ignoring the infrequent C load, the two
load instructions contribute equally, so predicted `sec/req` is
`(32 + 1) / 2 = 16.5`.

If both input operands of every FMA came from DRAM, each FMA would require
8 bytes for 2 FLOPs: arithmetic intensity `2/8 = 0.25 FLOP/byte`.
The bandwidth bound would be `1,555 * 0.25 = 388.75 GFLOP/s`, ignoring the
lower-order C traffic. At N = 4096, those input reads would total
`8*N^3 = 549,755,813,888 bytes`, about 549,756 decimal MB.

### Measured comparison — pending

Insert naive GFLOP/s and `DRAM_MB`. Compare with the bounds above and use
the measured traffic to explain operand reuse through caches. Do not infer
DRAM traffic directly from global-load sector counts.

## Part 2 — Coalesced

### Prediction before profiling

Now a warp has fixed row `r` and columns `c + l`, with `c` aligned to 32:

- A: `&A[r * N + k]`. All lanes access one word: one sector.
- B: `&B[k * N + c + l]`. The lanes access 32 consecutive floats:
  four sectors.

Predicted `sec/req = (1 + 4) / 2 = 2.5`. The predicted naive/coalesced
global-load sector ratio is `33/5 = 6.6`, ignoring C. This is a traffic
prediction, not a promise of a 6.6x speedup: caches, instruction execution,
and other bottlenecks also affect time.

### Measured comparison — pending

Compare both sector predictions with `ncu.txt`. Calculate speedup as
`naive_ms / coalesced_ms` from `ladder.txt`, and compare with
`naive_gld_Msec / coalesced_gld_Msec` from `ncu.txt`.

## Part 3 — Shared-memory tiling

### Predictions before profiling

For T = 32, each warp loads a contiguous row of each shared tile from
global memory. A and B each require four sectors per warp per slice.
Across 32 inner-loop iterations, coalesced requires `32 * (1 + 4) = 160`
sectors per warp; tiled requires `4 + 4 = 8`. Predicted
`coalesced_gld_Msec / tiled_gld_Msec = 160/8 = 20`, ignoring C traffic.
The ratio is not 32 because the sector cost of a tile-loading instruction
differs from that of the coalesced kernel's broadcast A load.

The tiled inner loop issues two source-level shared loads per FMA:
`As[ty][k]` and `Bs[k][tx]`. For T = 32, these are a broadcast and a
contiguous warp load, each requiring one wavefront. At one wavefront per
SM per cycle, the bound is 0.5 warp FMA instructions per cycle. Relative
to the A100's peak of two, this is `0.5/2 = 25%` of FP32 peak, about
4,873 GFLOP/s using the harness peak of 19,492 GFLOP/s.

### Barrier experiment — pending

Remove only the first barrier, rebuild, and save both outputs:

```sh
./sgemm check
compute-sanitizer --tool racecheck ./sgemm tiled 256 --once
```

Restore the first barrier, remove only the second, rebuild, and save both
outputs again. Restore both barriers and rebuild afterward.

Expected hazards: the first barrier prevents a shared read before another
thread has written the current tile (read-after-write). The second prevents
the next tile's writes from overwriting values another thread is still
reading (write-after-read). Record actual outcomes; a numerical check may
pass despite an unsynchronized access, whereas racecheck examines hazards.

### Measured comparison — pending

Insert the four experiment results and explain any disagreement between
correctness and racecheck. Compare the sector ratio with 20 and tiled's
measured `%peak` with the source-level 25% bound.

## Part 4 — Block shape and tile size

### Shape predictions before either sweep

Let `R` be registers per thread for `sgemm_coalesced_kernel`, taken from
`make ptxas` on LS6. **R = 32**, from the user-supplied compiler output saved
in `results/predictions/ptxas_predictions.txt`. The same compiled kernel
serves all four shapes. Using the README's three-limit model:

```text
b1024 = min(32, 2,  floor(65536 / (1024*32))) = min(32, 2, 2) = 2
b32   = min(32, 64, floor(65536 / (32*32)))   = min(32, 64, 64) = 32
b16   = min(32, 64, floor(65536 / (16*32)))   = min(32, 64, 128) = 32
```

Occupancy counts resident warps, including partially active warps; active
lanes are listed separately. The 32x32 shape reaches the register and warp
limits together. The other shapes reach the 32-block limit first.

| Shape | Threads/block | Warps/block | Active lanes/warp | Blocks/SM | Warps/SM | Occupancy | sec/req |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32x32 | 1024 | 32 | 32 | 2 | 64 | 100% | 2.5 |
| 32x1 | 32 | 1 | 32 | 32 | 32 | 50% | 2.5 |
| 4x4 | 16 | 1 | 16 | 32 | 32 | 50% | 2.5 |
| 1x32 | 32 | 1 | 32 | 32 | 32 | 50% | 16.5 |

Sector reasoning: both 32-wide shapes broadcast A and load 32 adjacent
B values (1 and 4 sectors). In 4x4, the 16 active lanes cover four rows
and four columns: A touches four row-separated sectors and B touches one
sector, giving `(4+1)/2 = 2.5`. In 1x32, A touches 32 row-separated sectors
and B broadcasts, giving 16.5. Its warp access pattern reproduces naive's.

Prediction: 32x32 is fastest among the coalesced shapes; 32x1 should beat
4x4 because its warps have twice as many useful lanes, even when their
occupancy and sectors/request match. The 1x32 shape should suffer from
uncoalesced A accesses. Final rankings and explanations require measured
GFLOP/s, occupancy, and sector counts.

### Tile-size predictions before the sweep

For each tile slice, the block loads `2*T*T` floats in `T*T/4` sectors
and performs `T*T*T` scalar FMAs. Therefore global-load sectors per scalar
FMA are `1/(4*T)`, ignoring C. Each warp executes two barriers per T warp
FMA instructions, giving `2/T` barrier instructions per warp FMA.
Shared storage is `2*T*T*4` bytes. These are source-level counts.

| T | Global sectors/scalar FMA | Barriers/warp FMA | Warps at each block barrier | Shared bytes/block |
|---|---:|---:|---:|---:|
| 8 | 1/32 | 1/4 | 2 | 512 |
| 16 | 1/64 | 1/8 | 8 | 2048 |
| 32 | 1/128 | 1/16 | 32 | 8192 |

Prediction: T = 32, then 16, then 8 in descending throughput, because
larger tiles reduce both global traffic and barrier frequency per FMA.
The competing effect is that larger blocks synchronize more warps and
allow fewer resident blocks; T = 32 need not win in practice. Keep this
prediction and explain any measured reversal using the profiler evidence.

### Measured comparison — pending

Compare predicted occupancy and sectors/request with `ncu_sweep.txt`.
Rank the four shapes by GFLOP/s, explain each loss to 32x32, compare 1x32
with naive, and explain 32x1 versus 4x4. Compare the tiled ranking with the
prediction above. Theoretical occupancy is a resource ceiling; achieved
occupancy can be lower.

## Part 5 — Provided 1-D register blocking

For each k, a thread loads one B value and eight A values from shared
memory, then performs eight FMAs. The source-level load count is
`(1+8)/8 = 9/8` per FMA, compared with tiled's 2. The reduction factor is
`2/(9/8) = 16/9`, about 1.78. Holding B in a register reuses it across
eight output rows.

The output tile also changes from 32x32 to 64x64, while threads per block
fall from 1024 to 512 and each thread holds more accumulators. These can
change global traffic, register use, and occupancy in addition to shared
load demand.

Pending: compare `tiled_ms / reg1d_ms` with 1.78, then cite an actual
`gld_Msec`, `regs`, or `occ%` difference showing another contributing
change. Do not assume timing scales exactly with the shared-load count.

## Part 6 — 2-D register blocking (pending implementation)

- Derive the source-level shared-load count and redo the throughput bound.
- Use measured registers/thread to calculate resident 256-thread blocks,
  warps, and occupancy; compare with achieved occupancy.
- Read GPU Speed Of Light Throughput, Occupancy, Memory Workload Analysis,
  and Warp State Statistics in `reg2d_full.txt`. Quote the busiest unit,
  dominant stall reason, and `wf/shld` to identify the limiter.

## Part 7 — One change (pending the Part 6 profile)

Before implementation, record one allowed change, the motivating metric
and baseline value, and predicted column changes with approximate sizes.
After implementation, compare reg2d and mine in both final tables and
explain the outcome. Justify a second optimization not taken using a
measured metric, its implementation/resource cost, and expected payoff.

## Final measurement tables (pending)

Include `ladder.txt`, `sweep.txt`, `ncu.txt`, and `ncu_sweep.txt` exactly as
printed by the final LS6 A100 run, preserving their environment context.
Use ladder times for speedups; compare profiler timings only with other
profiler timings because ncu locks clocks differently. Submit the batch
output, `reg2d_full.txt`, barrier outputs, source files, and AI log too.
