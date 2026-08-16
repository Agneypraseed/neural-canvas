# Neural Canvas Resume Copy

## Recommended project title

**Neural Canvas - Optimization-Based Neural Style Transfer**

## One-line description

Independent PyTorch implementation of Gatys-style neural transfer that optimizes an image against frozen VGG-19 content and multi-scale style features, exposed through a tested CLI and Gradio application.

## Resume bullets

- Implemented optimization-based neural style transfer in PyTorch with a frozen ImageNet-pretrained VGG-19, `relu4_2` content features, normalized multi-layer Gram statistics from `relu1_1` through `relu5_1`, total-variation regularization, and Adam updates to generated-image pixels only.
- Productized the pipeline as an installable CLI and resource-bounded Gradio app with CPU/CUDA/MPS selection, image validation, automated tests, Docker, and CI; measured one local cached-weight CPU run at **6.63 s end to end** for 25 Adam steps at a 128-pixel maximum edge (**4.68 s** in optimization).

## Technologies

Python, PyTorch, Torchvision, Pillow, NumPy, Gradio, Pytest, Ruff, setuptools, Docker, GitHub Actions

## Publication links

**GitHub:** [github.com/Agneypraseed/neural-canvas](https://github.com/Agneypraseed/neural-canvas) - expected repository URL; publication and public accessibility are pending verification.

**Live demo:** Pending permanent deployment and a successful public end-to-end smoke test. Do not add or claim a demo URL until that verification is complete.

## Evidence behind the performance claim

This is one measured local reference run, not an average, percentile, cross-device comparison, or latency guarantee. It ran on 2026-08-16 at `2026-08-16T15:13:14+02:00` under Windows 11 `10.0.26200`, with an `AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD` CPU, 12 logical processors, and PyTorch intra-op/inter-op thread counts of 6/6. The environment used Python 3.12.2, PyTorch 2.13.0+cpu, Torchvision 0.28.0+cpu, Pillow 11.3.0, and NumPy 2.5.1.

The source content and style images were each 384 x 256 pixels. The benchmark used the CPU, a 128-pixel configured maximum edge, 25 Adam steps, and official VGG-19 ImageNet weights that were already cached. It produced a 128 x 85 image. Measured timings were 1.9153 s for input hashing/decoding/resizing plus VGG construction, 4.6821 s for `run_style_transfer`, 0.0314 s for output encoding/hashing, and 6.6288 s total. The output SHA-256 was `96b3dbfc6a753ad12d6e8bbcee8432f4006acc29b584354e6186acb8c7f1b073`.

Current local verification on 2026-08-16 reported a fully passing test suite and a clean Ruff check. The bullet deliberately says "automated tests" rather than embedding the test count because that count changes as the suite evolves.

## Claims to avoid

Do not say that this project trained VGG-19, invented neural style transfer, uses a custom generative model, or performs real-time style transfer. Do not present the single local benchmark as typical production latency, and do not present GitHub or live-demo links as public until each is verified.
