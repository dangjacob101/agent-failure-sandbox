# Experiment source snapshots

`paper_pipeline_qwen15b/src/early_failure/` preserves the 18 Python source files
used for the completed experiment. Each file matches the recorded SHA-256 in
`results/paper_pipeline_qwen15b/source_snapshot.json`.

Active development belongs in the repository's top-level `src/`. The archived
artifact audit uses this snapshot so later implementation changes do not rewrite
the historical experiment. This is a source record, not a second package to edit.
