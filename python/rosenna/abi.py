"""The contract the two generated loaders share.

Both backends read the same `.rwt` file and must answer with the same status
code for the same defect, so the status table and the capacities `<model>_init`
declares for the file's table of contents live here rather than in either
emitter.
"""

STATUS_CODES = [
    (0, "success"),
    (1, "cannot open the weights file"),
    (2, "not a roseNNa weights file (bad magic)"),
    (3, "weights file version is not supported"),
    (4, "weights file dtype does not match this generated code"),
    (5, "weights file endianness does not match this machine"),
    (6, "weights file plan hash does not match this generated code"),
    (7, "weights file holds a tensor this model does not declare"),
    (8, "a name or rank in the weights file exceeds this model's capacity"),
    (9, "a read failed: the weights file is truncated or inconsistent"),
    (10, "device allocation or copy failed in init"),
    (11, "kernel launch failed"),
]

# Floors for the buffers `<model>_init` declares to parse the table of
# contents. The real capacity is the larger of the floor and what this plan's
# own tensors need -- a PyTorch initializer name is a dotted module path and
# routinely runs past any fixed size someone guessed at. Anything in the file
# that exceeds the declared capacity is rejected with status 8 rather than
# read, which is also what closes the file-parsing hole: `expected_hash` is
# compiled into the binary, so a crafted file can copy it verbatim and reach
# the name and rank reads with attacker-chosen sizes.
_NAME_FLOOR = 64
_RANK_FLOOR = 4


def status_code_comment(prefix: str, model: str) -> list:
    """The status table, rendered as comments for the generated source.

    A scientist who gets `status 6` back from a solver should not have to read
    this generator's Python to find out what it means, so the table is emitted
    next to the routine that returns it, in both languages.
    """
    lines = [f"{prefix} Status codes ({model}_init; infer_batch returns 0, 10 or 11):"]
    lines += [f"{prefix}  {code:>2}  {text}" for code, text in STATUS_CODES]
    return lines


def name_capacity(plan) -> int:
    return max(_NAME_FLOOR, max((len(w.name) for w in plan.weights), default=0))


def rank_capacity(plan) -> int:
    return max(_RANK_FLOOR, max((len(w.shape) for w in plan.weights), default=0))
