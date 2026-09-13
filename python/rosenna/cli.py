"""Command line interface: generate, verify, info."""
import argparse
import sys


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

    ver = sub.add_parser("verify", help="compile the generated code and compare against onnxruntime")
    ver.add_argument("model")
    ver.add_argument("--lang", choices=["fortran", "c", "both"], default="both")
    ver.add_argument("--precision", choices=["single", "double"], default=None)
    ver.add_argument("--cases", type=int, default=16, help="random inputs to compare")

    info = sub.add_parser("info", help="report ops, shapes and whether the model is supported")
    info.add_argument("model")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"rosenna {args.command}: not implemented yet", file=sys.stderr)
    return 2
