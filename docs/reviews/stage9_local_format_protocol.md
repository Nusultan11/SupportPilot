# Local verifier format correction diagnostic

Audience: the owner inspecting the local support diagnostic. Registered2026-11-16 after
the complete original object-format run and before any enum-format model calls.

The original run retains192 raw rows and its immutable freeze/source archive. It has
183 invalid outputs. A one-row smoke already showed a JSON root string rather than the
requested object. This is an observed interface problem, not evidence of correct semantics.

Compare a new explicit JSON root-enum contract with that retained object contract on the
same controlled fixtures. Change only the requested output representation and its matching
strict parser. Pin the same0.6B revision, unadapted NF4 weights, greedy seed20261113,
1024-input/64-output limits and15s soft backend budget. No label, evidence, threshold or
semantic rubric changes. Smoke one row before a once-only full run, retaining all errors.
The original instructions' semantic sentences stay unchanged; replace only the output-shape
sentence. A valid enum must be an entire JSON string with exactly supported, unsupported
or unknown. Bare text, code fences, objects and other enum values are invalid. No repair,
keyword extraction or mapping a failure to unknown is permitted.

This is exploratory execution-contract correction on exposed self-authored fixtures, not
fresh independent confirmation or a qualified judge selection. Report both complete runs,
invalid counts, fixture agreement and need-cluster uncertainty against always-unknown.
No customer facts are delivered. Human semantic/fluency metrics remain NOT_EVALUATED,
and a lower invalid rate alone cannot qualify the verifier or close the grounding gate.
