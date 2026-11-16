# Local constrained-verdict diagnostic

Audience: the owner reviewing the observed local model interface failure. Registered2026-11-16
before constrained inference. Original object full-run and free-enum smoke remain unchanged.
The enum smoke still emitted trailing punctuation after its JSON string; it remains invalid.

Use the existing LM Format Enforcer public-core adapter with a root JSON-string enum schema
for supported/unsupported/unknown. Add an explicitly optional backend output_schema that
requires constrained=True; absent override preserves the AssistantResponse schema. No new
dependency, model weights, customer-facing setting or qualified verifier is introduced.
Unit-check enum completion, EOS eligibility and rejection of trailing period before model calls.

Keep the enum-format prompt,192 controlled rows/eight needs, labels, pinned0.6B base revision,
NF4, seed20261113,1024 input/64 output tokens and15s soft budget unchanged. Freeze the new
decoder/config/runtime binding separately. One execution smoke (<1%) precedes the full run.
Retain all token-budget/time/schema/backend failures; never repair a raw output.

This is a combined representation/decoder intervention relative to the original object run.
The unforced-enum diagnostic has only one smoke observation; do not claim a matched full
causal estimate separating representation and decoding. No new independent confirmation,
semantic calibration or human verification is claimed on these already exposed fixtures.
Report strict validity and fixture agreement separately. A syntactically guaranteed enum can
still be wrong or falsely supported. No support verdict from this study authorizes customer
facts; all independent-human semantic metrics remain NOT_EVALUATED and default delivery closed.
