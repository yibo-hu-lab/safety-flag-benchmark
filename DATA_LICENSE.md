# Data license

The **derived data** in `data/` — our harmonized flag labels, the balanced item-ID
selections, the standardized prompt, and the per-item model outputs (confidence
signals and verdicts) — is released under **Creative Commons Attribution 4.0
International (CC-BY-4.0)**: https://creativecommons.org/licenses/by/4.0/

This repository does **not** redistribute any source-benchmark content. The
per-item files contain only source item IDs, our labels, and our model outputs;
they contain no benchmark prompts or text. The underlying items remain under the
license of their source benchmark (BeaverTails, XSTest, Ethics, WildGuard, Aegis,
ToxiChat); obtain the raw text from each source under its own terms. Provenance
(which source each item came from) is recorded per item via the `dataset` field.
