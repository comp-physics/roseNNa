"""The one exception type every pass raises, in a module that imports nothing.

frontend and fold both need it, and frontend calls fold while fold builds a
frontend Graph -- so the exception cannot live in either without a cycle.
"""


class UnsupportedModel(ValueError):
    """The model uses something roseNNa cannot generate code for."""
