# Third-party notices for the engine

The engine archives contain no weights or original Python sources.
The Rust implementation references these code components from
Haidass-Image-Instruct revision a23047cd6598ef11decc466bcdcc8853eb55a4ab:

- `runtime/qwenimage21_transformer.py`: Copyright 2026 Qwen-Image Team,
  The HuggingFace Team. All rights reserved.
- `runtime/qwenimage21_vae.py`: Copyright 2026 The Qwen Team and
  The HuggingFace Team. All rights reserved.

The two reference Python modules are licensed under the Apache License,
Version 2.0. Related Rust files are reimplemented/modified versions;
the reference license is in `licenses/Apache-2.0.txt`.
The prompt routing and spelling implementation is a Rust translation
of the Haidass Instruct project's routing and spelling code.
Rust dependencies retain their respective upstream terms.
This notice does not grant an Apache license to the whole engine,
nor does it license model weights.
