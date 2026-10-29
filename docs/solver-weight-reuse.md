# Reusing SE(3) fitting weights across target frames

For a fixed query frame, `fit_per_pixel_se3` computes query-to-reference
weights from that frame's normalized rigidity embeddings. The optional
spatial Gaussian also uses only query-frame pixel coordinates. Neither term
depends on the target frame.

The solver now loops over query-pixel chunks before target frames. Each
chunk computes its similarity logits, softmax/renormalized weights and
weighted source centroid once, then reuses them across all target frames.
Reference target points and their outer-product tables are prepared once
per call. Only one chunk's weights are retained; there is no full dense
query-by-reference weight cache.

Each query pixel still gets its own query-to-target SE(3), fitted from the
shared reference correspondences. Target centroids, covariances and Horn
fits still run separately for each target frame. Changing the query frame
requires new weights. Reference sampling, covariance arithmetic, Horn
eigensolves, jitter, conditioning checks, quaternion conversion and CPU
return values are unchanged.

## Measured results

Paired measurements on one NVIDIA RTX PRO 6000 Blackwell Server Edition
(96 GB), using the same physical GPU for both implementations and both
lengths. Baseline: upstream commit
`c39ef00f7198515d709f40c668688d4f869255ae`.

| Frames | Solver original / reused (s) | Solver speedup | Full pipeline original / reused (s) | Full speedup | Full peak allocated (GiB), both |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 4.483407 / 2.417518 | 1.855x | 6.134185 / 4.082011 | 1.503x | 8.630439 |
| 64 | 5.713270 / 3.069753 | 1.861x | 8.107350 / 5.463064 | 1.484x | 9.422223 |

The solver-only allocated peak increases from 7.912058 to 7.938884 GiB
at 50 frames, and from 8.104586 to 8.139801 GiB at 64 frames: an extra
27.47 / 36.06 MiB for the cached reference tables. Full-pipeline allocated
peaks remain unchanged. Allocated peaks include resident model weights;
the solver-only measurements also keep their cached network inputs resident.

Quaternion, translation and validity tensors were **bitwise identical**
between the original and optimized solver on these inputs, including source
identity transforms. All 6,914,880 / 8,890,560 non-source fits had finite
outputs. Rotation and translation differences were zero, with zero validity
mask changes. The same single fit at 64 frames failed conditioning in both
implementations; none failed at 50 frames. These are implementation-agreement
and cost measurements on one clip, not a GT accuracy evaluation or a claim
of universal bitwise equivalence.

### Protocol

- DAVIS 2017 `dance-twirl`: first 50 or 64 consecutive original JPEG frames,
  resized with OpenCV `INTER_AREA` to 504 x 280. Query frame 0; 141,120 dense
  source trajectories, all target frames retained.
- Released MoSE3 weights at Hugging Face revision
  `5280ca2ea4c8cf41ac174645fd3aada430af7f40`.
- FP32 weights, BF16 forward autocast, FP32 solver, TF32 disabled.
  PyTorch 2.11.0+cu128; checkpoint initialization excluded.
- Solver arguments: `query_idx=0`, `num_ref_pts=12000`, `temperature=0.01`,
  `sigma_px=0.0`, `chunk_size=4096`, `seed=0`.
- One full warmup and three synchronized timed repeats per variant, scope
  and frame count; tables report medians. CUDA allocator peaks reset before
  each timed call after releasing prior outputs and clearing unused cache.
- Solver-only variants consume identical cached 3D tracks and rigidity
  embeddings, and include the solver's CPU output copies.
- Full pipeline starts with decoded CPU RGB and ends with CPU tracks,
  visibility, full rigidity embeddings and SE(3). It includes upload,
  forward inference, fitting and output copies. Cached solver-only inputs
  are released before measuring this scope.
- Numerical comparisons and file writing are excluded from timings. The
  optional spatial Gaussian was preserved in code but was off in this run.

## Eager scalar-transfer and buffer-lifetime cleanup

Two additional changes retain the eager solver and its existing output API:

- Use Python scalar operands in the quaternion-sign `torch.where`, preserving
  the covariance dtype, instead of constructing and uploading two CUDA scalar
  tensors for every Horn call.
- Scale private logits and renormalize private weights in place. Release
  logits after softmax and weights after each source chunk. The second weight
  normalization is retained. Spatial-Gaussian arithmetic is unchanged; its
  coordinate temporaries are released after use.

The solver runs under `torch.no_grad()`, and these weight buffers are private
temporaries. Reference sampling, FP32 precision, covariance arithmetic, both
per-frame eigensolves, jitter, validity checks and quaternion conversion are
retained. No compiler, cross-frame Horn batching, packed GEMM or reduced-output
mode is introduced.

One 4096-by-12000 FP32 buffer occupies 187.5 MiB. Avoiding simultaneous old/new
buffers and releasing logits reduces local memory demand; the measurements
below report actual allocator peaks rather than estimating whole-call savings
from this buffer size.

The paired control is commit `5a8d2c1`, with both original `per_point_se3.py`
and `se3_fit.py` loaded from frozen snapshots. Each implementation uses the
same cached network outputs for solver-only timing and numerical comparison.
Full-pipeline timing retains the entire tracking/visibility/rigidity/SE(3)
CPU output contract. Inputs, precision, reference pool and repeats match the
protocol above. Separate profiler passes verify transfer/synchronization counts;
instrumented times are excluded from latency medians and overlap GPU execution.

In the separate 64-frame profiler passes, pageable H2D copies drop from
**4410 to 0** and `cudaStreamSynchronize` calls from **8823 to 4413**.
Both `eigvalsh` and `eigh` remain at 2205 calls each. The synchronization API
self CPU times are approximately 556.46 / 557.93 ms before / after cleanup;
fewer calls do not imply proportionally less waiting or a large speedup.
These instrumented durations overlap device execution and are not a partition
of unprofiled wall latency.

Quaternion, translation and validity outputs are bitwise identical on the
matched 50/64-frame inputs, including source identity tensors. All finite
masks agree and validity flips are zero. The same single 64-frame fit is
invalid in both implementations. This is agreement on the measured inputs,
not a universal bitwise or GT tracking-accuracy claim.

### Paired cleanup results

| Frames | Scope | Before / after seconds | Speedup | Before / after allocated peak GiB |
| ---: | --- | ---: | ---: | ---: |
| 50 | solver | 2.435165 / 2.392475 | 1.018x | 7.938884 / 7.755290 |
| 50 | full_pipeline | 4.079710 / 4.033410 | 1.011x | 8.630439 / 8.630439 |
| 64 | solver | 3.062922 / 3.005426 | 1.019x | 8.139801 / 7.956207 |
| 64 | full_pipeline | 5.940854 / 5.817977 | 1.021x | 9.422223 / 9.422223 |

Solver allocated peaks decrease by **188 MiB at both lengths**. Reserved
solver peaks fall from 9.244141 to 8.011719 GiB at 50 frames and from
9.542969 to 8.310547 GiB at 64 frames. Reserved memory is allocator
capacity rather than live tensor payload; these are different quantities.
Full-pipeline allocated peaks remain unchanged. Latency gains are modest
(about 2% for solver-only); the primary benefits are fewer small transfers
and lower solver memory, rather than a large end-to-end acceleration.

Every latency sample is uninstrumented; separate 64-frame profiler passes
precede that length's full-pipeline samples in the same process. All raw
repeats are retained in the analysis artifacts. Values are one-clip paired
measurements on one PRO 6000, not a hardware-independent speed guarantee.

## Direct Horn quaternion: 24-frame experiment

Experimental branch `codex/direct-horn-quaternion`, based on eager-cleanup
commit `326ea1b`. Horn already computes a canonicalized, normalized quaternion
before building R. Return that quaternion and export it directly, retaining
the unchanged R path for translation and finite/conditioning checks. This
removes the quaternion-to-R-to-quaternion round trip from output generation.
Both native eigensolves, weights, per-frame batch sizes and prior cleanups
remain unchanged. No compiler, frame batching or packed-GEMM change is added.

The test uses **exactly 24 consecutive frames**, 504 x 280, query frame 0,
141,120 source trajectories and **3,386,880 predicted positions**. Its RGB is
the first 24 frames of the saved consecutive DAVIS `dance-twirl` input used
for the 50/64-frame tests. The older 23-frame every-fourth-frame MP4 is a
different cohort. Baseline/candidate pairs share one physical RTX PRO 6000,
model weights, BF16 forward autocast, FP32 solver and disabled TF32. Solver
settings remain 12000 references, temperature 0.01, sigma 0, chunk 4096, seed 0.
One full warmup and three synchronized measured repeats per case:

| Scope | Baseline seconds | Direct quaternion seconds | Speedup | Baseline allocated GiB | Direct allocated GiB | Allocated change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solver only | 1.148190 | 0.845820 | 1.357x | 7.358998 | 7.359059 | +64 KiB |
| Tracking + rigidity + SE(3) | 1.750014 | 1.431839 | 1.222x | 7.522389 | 7.522450 | +64 KiB |

Time decreases 26.3% for the solver and 18.2% for the full pipeline. Reserved
peaks are unchanged: 7.568359 GiB solver-only and 8.697266 GiB full pipeline.
Allocated peaks include resident model weights/inputs. Full timing starts with
decoded CPU RGB and ends with CPU tracks, visibility, all rigidity embeddings
and SE(3); initialization and file writing are excluded.

The numerical comparison uses identical cached network outputs and all
3,245,760 non-source fits, which are finite and valid in both versions:

- Quaternion components are not bitwise identical. Sign-aligned max absolute
  difference: **4.1723e-7**.
- Rotation difference max / mean / p99: **4.5923e-5 / 3.0360e-7 / 6.9639e-6
  degrees**.
- Translation and validity tensors are **bitwise identical**; translation
  differences, validity flips and finite-mask mismatches are zero.

Direct export avoids FP32 rounding through R and the matrix-to-quaternion
conversion. These numbers describe this input and GPU, not universal bitwise
equivalence or GT tracking accuracy. The established eager-cleanup branch
`codex/reuse-se3-weights` remains at `326ea1b` while this change is evaluated.
