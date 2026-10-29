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
