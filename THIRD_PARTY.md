# Third-party components

Upstream weights and code retain their own licenses. The model sources consulted
on 2026-10-17 list Apache-2.0 for all four pinned checkpoints in config/models.json.
The download process retains model cards. Do not infer that other revisions or
other TimesFM checkpoints have identical terms. In particular, the upstream source
describes a different weight license for TimesFM 3.0.

The source repository does not vendor upstream model implementation code. Runtime
dependencies are installed separately. Do not assume small parameter counts imply
that all full-precision models fit simultaneously on a 16 GB machine.

Laya performance claims in its model card are task/checkpoint dependent. Its base
checkpoint is not established here as a generally accurate Jev replacement. This
repository uses it experimentally and records actual adapter failures.
