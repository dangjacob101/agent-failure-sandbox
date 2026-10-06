# Sources and attribution

This standalone sandbox adapts the continuation scoring, conformal state
labeling, and linear probing methodology of *From Actions to Understanding*:

- [Paper](https://arxiv.org/abs/2604.19775), Sections 4.1–4.3.
- [Paper code](https://github.com/trilokpadhi/interpretability-of-llm-agents),
  inspected at commit `0b041ccdd2e5de66b6300cbb14d20f06f226d295`.
- [interp-sandbox](https://github.com/skarnati20/interp-sandbox), inspected at
  commit `af85541efd8fb8e9ca0e60ff1ec8b88e34470b10`, informed the Qwen model family
  choice and MiniWoB DOM/action interface.

The sandbox imports neither upstream checkout. Their original model training
and vendored project trees are not included. MiniWoB, PyTorch, Transformers,
scikit-learn, Selenium, and other dependencies are installed separately through
`pyproject.toml` and `uv.lock`; model weights are downloaded separately from
[Qwen](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).

## License status

No repository-wide license has been selected for this sandbox. Attribution here
does not grant additional rights to third-party code, dependencies, or weights.

At the inspected revisions, neither upstream had a top-level license file. The
paper repository has an Apache license classifier in metadata identifying
FastChat/fschat, plus licenses within its vendored TextWorld tree; these are not
treated here as a license for its entire repository. The reference sandbox had
no tracked license file or license declaration found. This repository therefore
does not label either upstream, or the combined work, as MIT- or Apache-licensed.

Consult the respective projects for their own terms. The exact source revisions
are retained above so the methodological and interface provenance stays clear.
