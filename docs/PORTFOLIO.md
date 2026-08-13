# Portfolio and resume positioning

## Recommended project card

**Neural Canvas - Optimization-Based Style Transfer**

Reimplemented the Gatys neural style-transfer method in modern PyTorch. The system optimizes image pixels against multi-layer VGG-19 content and Gram-matrix style targets, with total-variation regularization, accelerator support, a CLI, and an interactive web demo.

**Tags:** PyTorch, Computer Vision, VGG-19, Optimization, Gradio, Docker

Link the card directly to this standalone repository and, once deployed, add a separate **Live demo** button. Do not link it to the broad `DL` coursework repository.

## Recommended resume bullets

Use two bullets, not a paragraph:

- Reimplemented optimization-based neural style transfer in PyTorch using frozen VGG-19 features, normalized Gram matrices, multi-layer style loss, and total-variation regularization.
- Productized the model with a tested Python package, CLI, Gradio demo, Docker image, device-aware execution, and reproducible configuration.

These bullets are honest and stronger than describing the old assignment as a standalone project: the repository now contains a modern independent reimplementation.

## What to replace

On the current two-page resume, replace **Car Detection for Autonomous Driving** if that entry still points to the guided YOLOv2 course assignment. The new project gives clearer evidence of current ownership because recruiters can run it, inspect tests, and see engineering decisions.

For a software-engineering role, keep Travel Aggregator first. For an ML/AI internship or working-student role, order projects as:

1. Preference-Aligned Chatbot with Gemma & DPO
2. Neural Canvas
3. Travel Aggregator

## Before publishing

1. Create a repository named `neural-canvas` (or update the URLs in `pyproject.toml` if you choose another slug).
2. Add three original demo images: content, style, and the actual generated result.
3. Record a 10-15 second screen capture of the Gradio controls and output.
4. Deploy the demo to a GPU-backed or CPU-compatible host and link it from the portfolio card.
5. Pin the repository on your GitHub profile.
