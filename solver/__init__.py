"""Legal Lane Solver package (L01-L05).

Public API:
    execute(tool_call: dict) -> dict         # dispatch a tool_call to the right solver
    roundtrip(sample: dict) -> bool          # NL-assisted roundtrip check on one sample
    validate_schema(tool_call: dict) -> bool # JSON Schema validation per lane

Lanes:
    L01 element_matching   (Z3 Bool AND/OR)
    L02 rule_application   (Z3 Implies)
    L03 subsumption        (Z3 Implies + Bool)
    L04 logic_judgment     (Z3 propositional validity)
    L05 calculation        (SymPy)
"""

from .executor import execute
from .validator import roundtrip, validate_schema

__all__ = ["execute", "roundtrip", "validate_schema"]
