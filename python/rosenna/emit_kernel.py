"""Render <name>_kernel.cu: the native device kernels for nvcc and hipcc.

One source in the CUDA subset hipcc accepts unchanged; every runtime call
goes through rosenna_rt.h. Two entry points:

infer_batch: one thread per point calling the header's infer, whose dense
layers are register-blocked (emit_c.GEMM_BLOCK). Staging the weights in
shared memory was tried and measured slower on an MI210; the bound was the
per-thread activation loads, which the blocking amortises.

infer_one: one sample, a launch per op with the thread index over the op's
output elements and the intermediate activations in static __device__
buffers. The form a model over a whole field needs; absent when the plan
has an LSTM. Those buffers are shared by every call, so infer_one orders
itself across streams with an event (status 12 if the event API fails).

Both are asynchronous on the caller's stream (ruling R5); the one transfer
this file makes (<name>_device_bind, file-loaded plans) is the plan step,
called from init. Validated on an A100 (nvcc 13.0, HPC SDK 25.11) and an
MI210 (ROCm 7.2.0, AFAR 23.2.1) by `rosenna gpu-gate`.
"""
from .emit_c import (KERNEL_TILE, _CTYPE, _c_weight_symbol, _device_bind, elem_call, elem_length,
                     field_buffer, has_infer_one, large_locals, scratch_symbols)
from .plan import Plan

# One block's threads for a fused run of small ops. An op whose output fits in
# this many threads can share a kernel with its neighbours, because
# __syncthreads() is a full barrier over a single block -- which is what makes
# the fusion sound: after the barrier, every element the next op reads has been
# written. An op larger than this keeps its own kernel and its own grid, where
# it gets the parallelism it needs; squeezing it into one block to fuse it
# would trade 5 us of launch for far more compute.
FUSE_THREADS = 256


def _fusion_runs(plan: Plan) -> list:
    """The op sequence split into runs to fuse and ops to launch alone.

    Returns [(fused, [(k, op), ...])]. A run of one is never worth fusing --
    it is the same single launch either way -- so it comes back as solo.
    """
    live = [(k, op) for k, op in enumerate(plan.ops) if op.kind != "alias"]
    runs, cur = [], []
    for k, op in live:
        if elem_length(op) <= FUSE_THREADS:
            cur.append((k, op))
            continue
        if cur:
            runs.append((len(cur) > 1, cur))
            cur = []
        runs.append((False, [(k, op)]))
    if cur:
        runs.append((len(cur) > 1, cur))
    return runs


def _emit_upload_device(plan: Plan, ctype: str) -> list:
    """The cuda/hip half of init: the device copies of a file-loaded model's weights.

    This lives here rather than in <name>.c because every line is a CUDA/HIP
    runtime call. <name>.c owns the host arrays and the OpenMP declare-target
    that puts them on the device for a per-point host; this owns the separate
    copies the native kernel reads. Both are filled from the same host arrays
    by the same init, so one archive now serves both call paths -- which is
    the whole point of the split.

    A repeated init frees the previous copies first (freeing a null pointer is
    a no-op in both runtimes). A failed allocation, copy or bind releases
    everything again and returns 10, so a later infer_batch refuses to launch
    (its null check) rather than reading an unfilled buffer or launching over
    a table that still holds the previous addresses. The bind is last: it
    publishes the new addresses to this translation unit's table.
    """
    m = plan.model
    syms = [_c_weight_symbol(m, w.symbol) for w in plan.weights]
    lines = [
        "/* The device copies of the weights. Declared extern in the header, so",
        "   any translation unit's device_bind_here can read them; defined here,",
        "   beside the runtime calls that fill them. */",
    ]
    lines += [f'extern "C" {ctype} *{s}_dev[ROSENNA_MAX_DEVICES] = {{0}};' for s in syms]
    lines += [
        "",
        "/* Per device: a multi-GPU host calls init (or upload_device) once per",
        "   device with that device current, and each device's __constant__ table",
        "   is filled with that device's pointers. Freeing one device's copies",
        "   must not disturb another's. */",
        f"static void {m}_release_device_at(int d) {{",
    ]
    for s in syms:
        lines += [f"    (void)ROSENNA_FREE({s}_dev[d]);", f"    {s}_dev[d] = 0;"]
    lines += [
        "}",
        "",
        f'extern "C" int {m}_upload_device(void) {{',
        "    int d = 0;",
        "    if (ROSENNA_GET_DEVICE(&d) != ROSENNA_OK) return 10;",
        "    if (d < 0 || d >= ROSENNA_MAX_DEVICES) return 13;",
        f"    {m}_release_device_at(d);",
    ]
    fail = f"{{ {m}_release_device_at(d); return 10; }}"
    for s in syms:
        lines += [
            f"    if (ROSENNA_MALLOC(&{s}_dev[d], sizeof {s}) != ROSENNA_OK) {fail}",
            f"    if (ROSENNA_MEMCPY_H2D({s}_dev[d], {s}, sizeof {s}) != ROSENNA_OK) {fail}",
        ]
    lines += [
        f"    if ({_device_bind(m)}() != 0) {fail}",
        "    return 0;",
        "}",
        "",
    ]
    return lines


def emit_kernel(plan: Plan) -> str:
    m = plan.model
    ctype = _CTYPE[plan.dtype]
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    # Which device is current, and has init run on THAT device? Both entry
    # points need it: the null check is per device now, so a host that
    # initialized device 0 and launched on device 1 gets status 10 instead of
    # a kernel reading another device's address.
    dev_prologue = ([
        "    int rdev = 0;",
        "    if (ROSENNA_GET_DEVICE(&rdev) != ROSENNA_OK) return 10;",
        "    if (rdev < 0 || rdev >= ROSENNA_MAX_DEVICES) return 13;",
    ] + [f"    if ({_c_weight_symbol(m, w.symbol)}_dev[rdev] == 0) return 10;"
         for w in plan.weights]
        if not plan.embed and plan.weights else [])
    dev_check = dev_prologue
    lines = [
        "/* Generated by rosenna. Do not edit. Build with nvcc or hipcc. */",
        '#include "rosenna_rt.h"',
        f'#include "{m}.h"',
        "",
        "#include <stddef.h>",
        "",
        f"#define ROSENNA_TILE {KERNEL_TILE}",
        f"#define ROSENNA_FUSE {FUSE_THREADS}",
        "",
    ]
    if not plan.embed and plan.weights:
        lines += _emit_upload_device(plan, ctype)
        lines += [
            "/* Plan step (controller ruling R5), called by init after it has made",
            "   the device copies: binds this translation unit's __constant__ table,",
            "   the one the kernel below reads, through the header's",
            f"   {_device_bind(m)}_here. Nothing else in this file transfers. */",
            f'extern "C" int {_device_bind(m)}(void) {{',
            f"    return {_device_bind(m)}_here();",
            "}",
            "",
        ]

    # --- infer_one ---
    if has_infer_one(plan):
        lines += ["/* infer_one's activations: device globals, referenced by name from the",
                  "   kernels (a __device__ variable's address is not a host value). */"]
        lines += [f"static __device__ {ctype} {field_buffer(m, sym)}[{plan.buffers[sym]}];"
                  for sym in scratch_symbols(plan)]
        lines.append("")
        runs = _fusion_runs(plan)
        for r, (fused, members) in enumerate(runs):
            if not fused:
                k, op = members[0]
                lines += [
                    f"static __global__ void {m}_k{k}(const {ctype} *__restrict__ x, {ctype} *__restrict__ y) {{",
                    "    const int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);",
                    f"    if (e < {elem_length(op)}) {elem_call(plan, k, op)};",
                    "}",
                ]
                continue
            sizes = ", ".join(str(elem_length(op)) for _, op in members)
            lines += [
                f"/* {len(members)} consecutive ops in one launch, output lengths {sizes}:",
                "   each fits in a block, and __syncthreads() between them is a full",
                "   barrier over that block, so every element the next op reads is",
                "   written. One block, so the barrier covers every thread that runs.",
                "   The barrier sits between the loops, never inside one -- a",
                "   __syncthreads() some threads skip is undefined. */",
                f"static __global__ void {m}_f{r}(const {ctype} *__restrict__ x, {ctype} *__restrict__ y) {{",
            ]
            for i, (k, op) in enumerate(members):
                # Strided over the block rather than one element per thread, so
                # the kernel is correct for ANY block size -- including a block
                # of one, which is how the stubbed host build in the tests
                # emulates a launch. That keeps FUSE_THREADS a performance
                # choice instead of a correctness precondition.
                lines += [
                    f"    for (int e = (int)threadIdx.x; e < {elem_length(op)}; e += (int)blockDim.x)",
                    f"        {elem_call(plan, k, op)};",
                ]
                if i + 1 < len(members):
                    lines.append("    __syncthreads();")
            lines.append("}")
        lines += [
            "",
            "/* The activation buffers above are shared by every infer_one call, so two",
            "   calls must not overlap. Calls on one stream are already ordered by the",
            "   stream itself; a call on a different stream is made to wait on an event",
            "   recorded after the previous call's last launch. Both the wait and the",
            "   record are asynchronous enqueues -- the loop path still never",
            "   synchronizes (ruling R5) -- and the event is created once, on the first",
            "   call, never again.",
            "",
            "   This orders the device work. It does not make infer_one callable from",
            "   several host threads at once: the two statics below are plain host",
            "   state with no lock. */",
            f"static ROSENNA_EVENT_T {m}_one_done[ROSENNA_MAX_DEVICES];",
            f"static int {m}_one_ready[ROSENNA_MAX_DEVICES] = {{0}};",
            f"static ROSENNA_STREAM_T {m}_one_stream[ROSENNA_MAX_DEVICES];",
            "",
            f'extern "C" int {m}_infer_one(const {ctype} *__restrict__ x, {ctype} *__restrict__ y, void *stream) {{',
            "    const ROSENNA_STREAM_T s = (ROSENNA_STREAM_T)stream;",
            *(dev_check or [
                "    int rdev = 0;",
                "    if (ROSENNA_GET_DEVICE(&rdev) != ROSENNA_OK) return 12;",
                "    if (rdev < 0 || rdev >= ROSENNA_MAX_DEVICES) return 13;",
            ]),
            "    /* The activation buffers are __device__, so they are already per",
            "       device; this ordering state is host-side and had to follow. */",
            f"    if (!{m}_one_ready[rdev]) {{",
            f"        if (ROSENNA_EVENT_CREATE(&{m}_one_done[rdev]) != ROSENNA_OK) return 12;",
            f"        {m}_one_ready[rdev] = 1;",
            f"    }} else if (s != {m}_one_stream[rdev]) {{",
            f"        if (ROSENNA_STREAM_WAIT_EVENT(s, {m}_one_done[rdev]) != ROSENNA_OK) return 12;",
            "    }",
            f"    {m}_one_stream[rdev] = s;",
        ]
        for r, (fused, members) in enumerate(runs):
            if fused:
                lines.append(f"    ROSENNA_LAUNCH({m}_f{r}, 1, ROSENNA_FUSE, s, x, y);")
            else:
                k, op = members[0]
                n = elem_length(op)
                lines.append(f"    ROSENNA_LAUNCH({m}_k{k}, ({n} + ROSENNA_TILE - 1) / ROSENNA_TILE, "
                             "ROSENNA_TILE, s, x, y);")
            lines.append("    if (ROSENNA_LAUNCH_STATUS() != ROSENNA_OK) return 11;")
        lines += [
            f"    if (ROSENNA_EVENT_RECORD({m}_one_done[rdev], s) != ROSENNA_OK) return 12;",
            "    return 0;", "}", ""]

    # --- infer_batch ---
    if large_locals(plan):
        lines += [
            f"/* {m}_infer's locals exceed a device thread's stack: infer_batch runs",
            "   infer_one per point, each a launch per op over the static buffers. */",
            f'extern "C" int {m}_infer_batch(int n, const {ctype} *__restrict__ x, {ctype} *__restrict__ y, void *stream) {{',
            "    int status = 0;",
            "    for (int p = 0; p < n && status == 0; ++p)",
            f"        status = {m}_infer_one(x + (size_t)p * {n_in}, y + (size_t)p * {n_out}, stream);",
            "    return status;",
            "}",
            "",
        ]
    else:
        lines += [
            "/* One thread per point; each calls the same inline body the host uses,",
            "   now instantiated as a __device__ function. */",
            f"static __global__ void {m}_kernel(int n, const {ctype} *__restrict__ x,",
            f"                                  {ctype} *__restrict__ y) {{",
            "    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);",
            "    if (p >= n) return;",
            f"    {m}_infer(x + (size_t)p * {n_in}, y + (size_t)p * {n_out});",
            "}",
            "",
        ]
        lines += [
            f'extern "C" int {m}_infer_batch(int n, const {ctype} *__restrict__ x, {ctype} *__restrict__ y, void *stream) {{',
            "    if (n <= 0) return 0;",
            "    const ROSENNA_STREAM_T s = (ROSENNA_STREAM_T)stream;",
            # A null device copy means init never ran or failed: status 10 rather
            # than a device fault. Reading the host globals is not a transfer.
            *dev_check,
            "    const int grid = (n + ROSENNA_TILE - 1) / ROSENNA_TILE;",
            f"    ROSENNA_LAUNCH({m}_kernel, grid, ROSENNA_TILE, s, n, x, y);",
            "    /* GetLastError, not a sync (ruling R5): a bad configuration or stream",
            "       is reported now; asynchronous faults surface at the caller's sync. */",
            "    if (ROSENNA_LAUNCH_STATUS() != ROSENNA_OK) return 11;",
            "    return 0;",
            "}",
            "",
        ]

    lines += [
        "/* The one synchronization in this file, and only when the caller asks:",
        "   a host with no stream of its own (an OpenMP host) waits here before",
        "   its next target region reads y. */",
        f'extern "C" int {m}_sync(void *stream) {{',
        "    return ROSENNA_SYNC((ROSENNA_STREAM_T)stream) == ROSENNA_OK ? 0 : 11;",
        "}",
        "",
    ]
    return "\n".join(lines)
