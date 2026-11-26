# Haidass Image model files

The engine does not contain or automatically download weights. The planned
engine tag is `engine-v0.1.0`; the planned compatible weight tag is
`weights-v0.1.0`. Neither tag name by itself indicates that binary assets
have been published. The engine is built and tested independently of model
downloads by CI.

## Source revisions used for conversion

- [DALabCommunity/Haidass-Image-Instruct](https://huggingface.co/DALabCommunity/Haidass-Image-Instruct/tree/a23047cd6598ef11decc466bcdcc8853eb55a4ab):
  `dit.safetensors`, `dit_norm_precision.safetensors`,
  `vae/diffusion_pytorch_model.safetensors` and
  `writing_lora/adapter_model.safetensors`.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca):
  the text encoder and tokenizer, licensed under Apache-2.0.
- The VAE file matches [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3/vae)
  (SHA-256 `a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`).
  The [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)
  applies to that VAE. It permits non-commercial research/evaluation only;
  separate permission is needed for commercial use. The Instruct snapshot
  does not declare a redistribution license. A public model URL is not a grant.

## Artifacts

The converter produces a main .himg bundle and an optional, pre-merged
writing bundle for each quantization. Keep the main file and its matching
writing file together if you want quoted-text routing. Q8 was the better
default in the documented Windows measurements. See the README for the
conversion command. Do not commit model files into Git history.
