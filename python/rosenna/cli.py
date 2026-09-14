"""Command line interface: generate, verify, info."""
import argparse
import sys
import tempfile
from pathlib import Path

from google.protobuf.message import DecodeError

from .emit_c import emit_c, emit_c_recipe
from .emit_fortran import emit_fortran, emit_fortran_recipe
from .emit_kernel import emit_kernel
from .gate import run_gate
from .rt_header import rt_header
from .frontend import UnsupportedModel, load_graph
from .plan import build_plan, validate_model_name
from .verify import VerificationError, verify_model
from .weights import write_weights

_PRECISION_TO_DTYPE = {"single": "f32", "double": "f64"}


def _dtype_from_precision(precision: str | None) -> str | None:
    return _PRECISION_TO_DTYPE.get(precision) if precision else None


def _add_embed_flags(sub: argparse.ArgumentParser) -> None:
    group = sub.add_mutually_exclusive_group()
    group.add_argument("--embed-weights", dest="embed", action="store_true", default=None,
                       help="embed weights as constants in the header, regardless of size "
                            "(default: embed automatically below EMBED_THRESHOLD parameters)")
    group.add_argument("--no-embed", dest="embed", action="store_false",
                       help="always load weights from a .rwt file at runtime")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rosenna", description="ONNX to Fortran/C inference code")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="emit inference sources and a weights file")
    gen.add_argument("model")
    gen.add_argument("--lang", choices=["fortran", "c", "both"], default="both")
    gen.add_argument("--precision", choices=["single", "double"], default=None,
                     help="default: the model's own dtype")
    gen.add_argument("--out", default=".")
    gen.add_argument("--name", default=None, help="symbol prefix; default: the model file stem")
    _add_embed_flags(gen)

    ver = sub.add_parser("verify", help="compile the generated code and compare against onnxruntime")
    ver.add_argument("model")
    ver.add_argument("--lang", choices=["fortran", "c", "both"], default="both")
    ver.add_argument("--precision", choices=["single", "double"], default=None)
    ver.add_argument("--cases", type=int, default=16, help="random inputs to compare")
    _add_embed_flags(ver)

    info = sub.add_parser("info", help="report ops, shapes and whether the model is supported")
    info.add_argument("model")

    gate = sub.add_parser(
        "gpu-gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="build and run the device-library validation harnesses on a GPU machine "
             "(gemm_big, embedded and file-loaded, both languages, three harnesses); "
             "writes gate-report.md",
        epilog="""\
Harnesses 1 and 2 (the per-point C and Fortran hosts) always need a HOST
compiler capable of OpenMP target offload, regardless of --backend: the
per-point infer() call always goes through the host compiler's own offload
region, never through --devcc. Concrete pairings:

  --cc nvc      --fc nvfortran --flags "-mp=gpu -gpu=cc80"              --backend cuda --devcc nvcc
  --cc amdclang --fc amdflang  --flags "-fopenmp --offload-arch=gfx90a" --backend hip  --devcc hipcc
  --cc gcc      --fc gfortran  --flags -fopenmp                        --backend omp  --host-fallback   (no GPU)
""")
    gate.add_argument("--cc", required=True, help="host C compiler")
    gate.add_argument("--fc", required=True, help="host Fortran compiler")
    # A value here that itself starts with '-' (e.g. -fopenmp, or a
    # multi-flag string like "-mp=gpu -gpu=cc80") is handled by
    # _join_dash_valued_options below, not by argparse's own parsing.
    gate.add_argument("--flags", default="", help="host offload flags, e.g. -fopenmp")
    gate.add_argument("--backend", choices=["cuda", "hip", "omp"], required=True,
                      help="which infer_batch implementation to build and exercise")
    gate.add_argument("--devcc", default=None,
                      help="device compiler for --backend cuda|hip, and the link driver "
                           "for the file-loaded C harnesses there (ruling R14); NOT "
                           "required -- default: nvcc for cuda, hipcc for hip")
    # Same dash-valued handling as --flags; see the comment above.
    gate.add_argument("--devflags", default="", help="device compiler flags")
    gate.add_argument("--out", default=".", help="directory for generated sources and gate-report.md")
    gate.add_argument("--host-fallback", action="store_true",
                      help="drop the OMP_TARGET_OFFLOAD=MANDATORY requirement so the omp "
                           "backend can be exercised on a machine with no accelerator")
    return p


def _describe_ops(graph) -> list:
    def shape_of(name: str) -> str:
        if name in graph.values:
            return str(tuple(graph.values[name].shape))
        if name in graph.initializers:
            return str(tuple(graph.initializers[name].shape))
        return "?"

    lines = []
    for node in graph.nodes:
        ins = ", ".join(f"{n}{shape_of(n)}" for n in node.inputs)
        outs = ", ".join(f"{n}{shape_of(n)}" for n in node.outputs)
        lines.append(f"{node.op} {node.name}: ({ins}) -> ({outs})")
    return lines


def _cmd_generate(args) -> int:
    graph = load_graph(args.model, name=args.name)
    plan = build_plan(graph, dtype=_dtype_from_precision(args.precision), embed=args.embed)
    validate_model_name(plan.model)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    name = plan.model
    langs = ["fortran", "c"] if args.lang == "both" else [args.lang]

    written = []
    if "fortran" in langs:
        f90_path = outdir / f"{name}_model.f90"
        fmk_path = outdir / f"{name}_fortran.mk"
        f90_path.write_text(emit_fortran(plan))
        fmk_path.write_text(emit_fortran_recipe(plan))
        written += [f90_path, fmk_path]
    if "c" in langs:
        source, header = emit_c(plan)
        recipe = emit_c_recipe(plan)
        c_path = outdir / f"{name}.c"
        h_path = outdir / f"{name}.h"
        mk_path = outdir / f"{name}.mk"
        cu_path = outdir / f"{name}_kernel.cu"
        rt_path = outdir / "rosenna_rt.h"
        c_path.write_text(source)
        h_path.write_text(header)
        mk_path.write_text(recipe)
        # The native batched kernel and its runtime map: built only when the
        # recipe runs with ROSENNA_BACKEND=cuda|hip, inert otherwise.
        cu_path.write_text(emit_kernel(plan))
        rt_path.write_text(rt_header())
        written += [c_path, h_path, mk_path, cu_path, rt_path]

    # An embedded plan has no weights file to write in either language: every
    # weight is already a `parameter`/ROSENNA_CONST array baked into the
    # generated source (controller ruling R3, flipped by Task 4: Fortran now
    # embeds by default too, so this no longer depends on which languages
    # were requested).
    if plan.embed:
        print(f"embedded weights ({plan.n_params} parameters)")
    else:
        rwt_path = outdir / f"{name}.rwt"
        write_weights(plan, graph, rwt_path)
        written.append(rwt_path)

    for path in written:
        print(path)
    return 0


def _cmd_verify(args) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        results = verify_model(args.model, args.lang, _dtype_from_precision(args.precision),
                                args.cases, workdir, embed=args.embed)
    all_ok = True
    for r in results:
        status = "ok" if r.ok else "FAIL"
        print(f"{r.lang} {r.cases} {r.max_abs:.6e} {r.max_rel:.6e} {status}")
        all_ok = all_ok and r.ok
    return 0 if all_ok else 1


def _cmd_gate(args) -> int:
    return run_gate(cc=args.cc, fc=args.fc, flags=args.flags, backend=args.backend,
                    devcc=args.devcc, devflags=args.devflags, out=args.out,
                    host_fallback=args.host_fallback)


def _cmd_info(args) -> int:
    graph = load_graph(args.model)
    for line in _describe_ops(graph):
        print(line)
    try:
        build_plan(graph)
    except UnsupportedModel as e:
        print(str(e))
        return 1
    print("supported")
    return 0


_DASH_VALUED_OPTIONS = ("--flags", "--devflags")


def _join_dash_valued_options(argv: list[str]) -> list[str]:
    """Let --flags/--devflags take a value that itself starts with '-' (e.g. -fopenmp).

    argparse treats any token starting with a prefix character as a
    candidate option string, even one no parser here defines, so `--flags
    -fopenmp` (two argv entries) fails with "expected one argument" --
    exactly the invocation shape `rosenna gpu-gate` needs for real compiler
    flags. Folding it into one `--flags=-fopenmp` entry first sidesteps
    argparse's option-likely-string heuristic entirely; the `=` form always
    works because argparse never re-examines what follows `=`.
    """
    out = list(argv)
    i = 0
    while i < len(out) - 1:
        if out[i] in _DASH_VALUED_OPTIONS and out[i + 1].startswith("-"):
            out[i:i + 2] = [f"{out[i]}={out[i + 1]}"]
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    argv = _join_dash_valued_options(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "gpu-gate":
            return _cmd_gate(args)
        raise AssertionError(f"unhandled command {args.command!r}")
    except UnsupportedModel as e:
        print(f"rosenna: {e}", file=sys.stderr)
        return 1
    except VerificationError as e:
        print(f"rosenna: {e}", file=sys.stderr)
        return 1
    except (OSError, DecodeError) as e:
        # A mistyped path (FileNotFoundError, an OSError) and a file that is not
        # a protobuf at all (DecodeError) are first-run mistakes, not bugs; they
        # belong on stderr as one sentence, not as a traceback.
        print(f"rosenna: {e}", file=sys.stderr)
        return 1
