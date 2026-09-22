# Work unit B — data/reference boundary and directional measurements

Reverified dev `fb988279` and the single active branch `fix/npo-adapter-qra-metrics`.
PR #8 is merged; stale live statements reconciled, historical entries preserved.
Quarantined rejected configurations without changing their original bytes.

The real tokenizer test caught Transformers 5 returning a mapping where upstream
expects token IDs. Explicit return_dict=False repairs that interface. Pinned
upstream collation also masks genuine EOT because pad==EOS; the recorded adaptation
uses length-based attention. No author tagged-prompt parity is claimed.

The researcher selected paper arithmetic Truth Ratio, retaining the geometric
upstream variant as a named diagnostic. Behavior tests cover answer log probability,
ROUGE-L recall, generated exact match, utility components, KS D/p and paired bootstrap.
All scientific decisions stay null. CPU acceptance: 269 passed, four unchanged skips,
ten GPU/model exclusions and historical SSIUU xfail. See immutable CPU evidence.

No model experiment or training has run. The bounded Mother child selects four fixed
rows, validates frozen FP16 forwards and records raw outputs. INT8/NF4 model loading
is deferred: source inspection identifies a 0.46.1 minimum bitsandbytes in Transformers
5 versus the validated 0.45.5 overlay. Preserve accepted environment evidence and
resolve this exact integration only before a precision pilot. No blanket dependency
redesign or global package change was performed.

The first bounded GPU submission passed (Mother version 2), run
`20260922-101703-ce77656bee84`, code `318e523a`. All four candidate/reference input
hashes and sequence losses matched; no gradient graph or optimizer step. Peak
allocation 5,120,751,616 bytes and reservation 5,249,171,456 bytes on one T4.
Independent re-derivation verified 59 sealed raw members and the reported metric
calculations. Complete private Kaggle/HF archives contain 68 verified members.
One archive creation title was too long; corrected the metadata before successful
creation without another kernel run. Initial error preserved. RG-006 resolved;
RG-005 and RG-001/002/003 remain blocked. PR #9 is open and must not be merged in
this work unit. The next work unit is directional precision integration and
scientific decision-protocol review; no corrected pilot yet.
