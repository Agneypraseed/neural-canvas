# From assignment artifact to independent project

The original artifact was useful learning work, but it was not a strong portfolio endpoint:

| Original artifact | Neural Canvas |
| --- | --- |
| Guided Coursera notebook | Independent package organized around the research method |
| TensorFlow 1 sessions and graph mutation | Current PyTorch eager-mode optimization |
| Missing `nst_utils.py`, images, and MatConvNet weights | Installable dependencies and official torchvision weights |
| Fixed course inputs and hyperparameters | CLI and UI controls for arbitrary images |
| No automated tests | Offline loss, image, config, and optimization-loop tests |
| Linked to an entire coursework dump | Standalone repository with focused documentation |
| No deployment surface | Gradio app and Docker image |

## Attribution language

Use: **“Independent PyTorch reimplementation of Gatys et al.'s neural style-transfer method.”**

Avoid: **“Invented a neural style-transfer model from scratch.”** The algorithm and VGG feature representation come from published research, and good project communication should make that provenance clear.
