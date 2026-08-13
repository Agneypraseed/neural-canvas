"""Interactive Gradio demo for Neural Canvas."""

# The embedded CSS and HTML are intentionally kept readable in their rendered form.
# ruff: noqa: E501

from functools import lru_cache
from html import escape
from time import perf_counter

import gradio as gr
from PIL import Image

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.engine import LossSnapshot, resolve_device, run_style_transfer
from neural_style_transfer.image_io import (
    pil_to_tensor,
    preserve_content_colors,
    resize_long_edge,
    tensor_to_pil,
)
from neural_style_transfer.model import VGG19FeatureExtractor

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink: #20201e;
  --muted: #6e6a62;
  --line: #ded9d0;
  --paper: #f5f1ea;
  --panel: #fbfaf7;
  --orange: #e96942;
  --orange-dark: #c94d2b;
  --blue: #173c4d;
  --blue-soft: #dce9eb;
  --green: #2e6a5b;
}

body, .gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 0 30px 52px !important;
}

footer { display: none !important; }

#site-header {
  align-items: center;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  padding: 22px 0 18px;
}

.brand-lockup { align-items: center; display: flex; gap: 11px; }
.brand-mark {
  align-items: center;
  background: var(--orange);
  border-radius: 9px;
  color: white;
  display: flex;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 16px;
  font-weight: 700;
  height: 31px;
  justify-content: center;
  letter-spacing: -0.05em;
  width: 31px;
}
.brand-name { color: var(--ink) !important; font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 700; letter-spacing: -0.03em; }
.header-meta {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  letter-spacing: .02em;
  text-transform: uppercase;
}

#hero { padding: 66px 0 48px; }
.hero-kicker, .section-kicker {
  color: var(--orange-dark);
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.hero-title {
  color: var(--ink) !important;
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(42px, 6vw, 76px);
  font-weight: 600;
  letter-spacing: -0.075em;
  line-height: .98;
  margin: 16px 0 22px;
  max-width: 760px;
}
.hero-title em { color: var(--orange); font-style: normal; }
.hero-copy { color: var(--muted); font-size: 17px; line-height: 1.6; margin: 0; max-width: 595px; }
.hero-aside {
  border-left: 1px solid var(--line);
  margin-top: 24px;
  padding: 4px 0 4px 25px;
}
.hero-aside p { color: var(--muted); font-size: 13px; line-height: 1.55; margin: 8px 0 0; max-width: 245px; }
.hero-aside strong { color: var(--ink); font-weight: 600; }

#workspace { align-items: stretch; gap: 18px; }
#inputs-panel, #result-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 14px 35px rgba(49, 43, 34, .045);
  padding: 25px;
}
#inputs-panel { min-width: 0; }
#result-panel { min-width: 0; }
.section-heading {
  align-items: baseline;
  display: flex;
  justify-content: space-between;
  margin-bottom: 18px;
}
.section-title { color: var(--ink) !important; font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 600; letter-spacing: -.045em; margin: 4px 0 0; }
.section-note { color: var(--muted); font-size: 12px; }

.image-label { color: var(--ink); font-size: 13px; font-weight: 600; margin: 0 0 8px; }
.image-label span { color: var(--orange-dark); font-family: 'DM Mono', monospace; font-size: 10px; margin-right: 6px; }
.image-help { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 8px 0 0; }

.upload-box {
  background: #f3f0eb !important;
  border: 1px dashed #c9c2b8 !important;
  border-radius: 12px !important;
  min-height: 211px !important;
  transition: border-color .2s ease, background .2s ease;
}
.upload-box:hover { background: #eeeae3 !important; border-color: var(--orange) !important; }
.upload-box .wrap { min-height: 211px !important; }
.upload-box .wrap { color: var(--muted) !important; }
.upload-box .or { color: #9b958c !important; }
.upload-box .icon-wrap { color: var(--orange) !important; }
.upload-box .upload-text { color: var(--muted) !important; font-size: 12px !important; }

#examples-wrap { margin: 22px 0 18px; }
#examples-wrap .label { color: var(--muted) !important; font-size: 12px !important; }
#examples-wrap .table-wrap, #examples-wrap table, #examples-wrap td { background: #f3f0eb !important; color: var(--ink) !important; }
#examples-wrap .tr-body { background: #f3f0eb !important; }
#examples-wrap button { border-radius: 9px !important; }

.controls {
  border-top: 1px solid var(--line);
  margin-top: 22px;
  padding-top: 18px;
}
.controls .label-wrap span, .controls label { color: var(--ink) !important; font-size: 12px !important; }
.controls label > span { color: var(--ink) !important; }
.controls .info { color: var(--muted) !important; font-size: 11px !important; }
.controls .form, .controls .block, .controls .wrap, .controls .tab-like-container { background: transparent !important; }
.controls .info-text, .controls .min_value, .controls .max_value { color: var(--muted) !important; }
.controls input[type='number'] { background: #f3f0eb !important; border: 1px solid var(--line) !important; color: var(--ink) !important; }
.controls .reset-button { background: transparent !important; color: var(--muted) !important; }
.controls input[type='range'] { accent-color: var(--orange) !important; }
.controls .wrap { margin-bottom: 13px; }

.action-row { align-items: center; gap: 12px; margin-top: 6px; }
#run-button {
  background: var(--orange) !important;
  border: 1px solid var(--orange) !important;
  border-radius: 10px !important;
  color: white !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  min-height: 46px !important;
  transition: background .2s ease, transform .2s ease;
}
#run-button:hover { background: var(--orange-dark) !important; transform: translateY(-1px); }
#clear-button { background: #f3f0eb !important; border: 1px solid var(--line) !important; border-radius: 10px !important; color: var(--muted) !important; font-size: 12px !important; min-height: 46px !important; }
.run-hint { color: var(--muted); font-size: 11px; line-height: 1.4; margin: 11px 0 0; }

.output-head { align-items: flex-end; display: flex; justify-content: space-between; margin-bottom: 15px; }
.output-head .section-title { margin-top: 4px; }
.live-pill {
  align-items: center;
  background: #edf5f0;
  border: 1px solid #d2e6dc;
  border-radius: 999px;
  color: var(--green);
  display: inline-flex;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  gap: 6px;
  padding: 7px 10px;
  text-transform: uppercase;
}
.live-dot { background: #4d9b77; border-radius: 50%; height: 6px; width: 6px; }
.output-box { background: #ebe7df !important; border: 0 !important; border-radius: 12px !important; overflow: hidden !important; }
.output-box .wrap { min-height: 470px !important; }
.output-box .wrap { color: #908a80 !important; }
.output-box img { object-fit: contain !important; }
.output-placeholder { align-items: center; background: #ebe7df; border-radius: 12px; color: #908a80; display: flex; flex-direction: column; justify-content: center; min-height: 470px; text-align: center; }
.output-placeholder .placeholder-icon { color: var(--orange); font-size: 34px; margin-bottom: 10px; }
.output-placeholder p { font-size: 13px; margin: 0; }
.output-placeholder small { color: #aaa39a; font-size: 11px; margin-top: 6px; }

.run-summary { margin-top: 15px; }
.run-summary.empty { background: #f5f2ed; border: 1px solid #e6e0d8; border-radius: 11px; color: var(--muted); font-size: 12px; padding: 13px 15px; }
.summary-top { align-items: center; display: flex; justify-content: space-between; margin-bottom: 11px; }
.summary-title { color: var(--green); font-size: 13px; font-weight: 600; }
.summary-time { color: var(--muted); font-family: 'DM Mono', monospace; font-size: 10px; }
.metrics { display: grid; gap: 8px; grid-template-columns: repeat(4, 1fr); }
.metric { background: #f5f2ed; border: 1px solid #e6e0d8; border-radius: 9px; padding: 10px; }
.metric-label { color: var(--muted); display: block; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .06em; margin-bottom: 6px; text-transform: uppercase; }
.metric-value { color: var(--ink); display: block; font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.summary-foot { color: var(--muted); font-size: 11px; line-height: 1.45; margin: 12px 0 0; }

#method { border-top: 1px solid var(--line); margin-top: 52px; padding-top: 31px; }
.method-title { color: var(--ink) !important; font-family: 'Space Grotesk', sans-serif; font-size: 24px; letter-spacing: -.045em; margin: 7px 0 24px; }
.method-grid { display: grid; gap: 13px; grid-template-columns: repeat(3, 1fr); }
.method-card { background: rgba(251, 250, 247, .66); border: 1px solid var(--line); border-radius: 13px; padding: 18px; }
.method-number { color: var(--orange); font-family: 'DM Mono', monospace; font-size: 11px; }
.method-card h3 { color: var(--ink) !important; font-family: 'Space Grotesk', sans-serif; font-size: 16px; letter-spacing: -.03em; margin: 15px 0 7px; }
.method-card p { color: var(--muted); font-size: 12px; line-height: 1.55; margin: 0; }

#site-footer { align-items: center; border-top: 1px solid var(--line); color: var(--muted); display: flex; font-family: 'DM Mono', monospace; font-size: 10px; justify-content: space-between; margin-top: 45px; padding-top: 18px; text-transform: uppercase; }

@media (max-width: 800px) {
  .gradio-container { padding: 0 16px 35px !important; }
  #hero { padding: 42px 0 30px; }
  #workspace { flex-direction: column !important; }
  #inputs-panel, #result-panel { padding: 18px; }
  .output-box .wrap, .output-placeholder { min-height: 320px !important; }
  .method-grid { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .header-meta { display: none; }
}
"""


@lru_cache(maxsize=3)
def _feature_extractor(device_name: str) -> VGG19FeatureExtractor:
    config = StyleTransferConfig(device=device_name)
    layers = {config.content_layer, *(name for name, _ in config.style_layers)}
    return VGG19FeatureExtractor(layers).to(resolve_device(device_name)).eval()


def _summary_html(
    result_device: str,
    final: LossSnapshot,
    elapsed: float,
    steps: int,
    image_size: int,
) -> str:
    safe_device = escape(result_device.upper())
    return f"""
    <div class="summary-top">
      <span class="summary-title">Render complete</span>
      <span class="summary-time">{elapsed:.1f}s wall time</span>
    </div>
    <div class="metrics">
      <div class="metric"><span class="metric-label">Device</span><span class="metric-value">{safe_device}</span></div>
      <div class="metric"><span class="metric-label">Steps</span><span class="metric-value">{steps:,}</span></div>
      <div class="metric"><span class="metric-label">Edge</span><span class="metric-value">{image_size}px</span></div>
      <div class="metric"><span class="metric-label">Total loss</span><span class="metric-value">{final.total:.5f}</span></div>
    </div>
    <p class="summary-foot">Content {final.content:.5f} &nbsp;·&nbsp; Style {final.style:.5f} &nbsp;·&nbsp; TV {final.total_variation:.5f}<br>Pixels were optimized against frozen VGG-19 features; model weights stayed unchanged.</p>
    """


def stylize(
    content_image: Image.Image | None,
    style_image: Image.Image | None,
    steps: int,
    image_size: int,
    style_exponent: float,
    preserve_color: bool,
    progress: gr.Progress | None = None,
) -> tuple[Image.Image, str]:
    """Run a style-transfer render and return the image plus run metadata."""

    if content_image is None or style_image is None:
        raise gr.Error("Add both a content image and a style reference to start a render.")

    started_at = perf_counter()
    progress = progress or gr.Progress()
    config = StyleTransferConfig(
        image_size=int(image_size),
        steps=int(steps),
        style_weight=10 ** float(style_exponent),
        progress_interval=max(1, int(steps) // 50),
    )
    resized_content = resize_long_edge(content_image, config.image_size)
    resized_style = resize_long_edge(style_image, config.image_size)

    def update_progress(snapshot: LossSnapshot, _image: object) -> None:
        progress(
            (snapshot.step, config.steps),
            desc=f"Optimizing pixels · step {snapshot.step}/{config.steps}",
        )

    extractor = _feature_extractor(config.device)
    result = run_style_transfer(
        pil_to_tensor(resized_content),
        pil_to_tensor(resized_style),
        config,
        feature_extractor=extractor,
        callback=update_progress,
    )
    output = tensor_to_pil(result.image)
    if preserve_color:
        output = preserve_content_colors(output, resized_content)

    final = result.history[-1]
    summary = _summary_html(
        result.device,
        final,
        perf_counter() - started_at,
        config.steps,
        config.image_size,
    )
    return output, summary


def _empty_summary() -> str:
    return '<div class="run-summary empty">Your render details will appear here after the first run.</div>'


def _reset_workspace() -> tuple[None, None, None, str]:
    return None, None, None, _empty_summary()


with gr.Blocks(title="Neural Canvas · Neural style transfer", fill_width=True) as demo:
    gr.HTML(
        """
        <header id="site-header">
          <div class="brand-lockup"><div class="brand-mark">N</div><div class="brand-name">Neural Canvas</div></div>
          <div class="header-meta">optimization-based style transfer &nbsp; / &nbsp; v0.1</div>
        </header>
        """
    )

    with gr.Row(elem_id="hero"):
        with gr.Column(scale=7):
            gr.HTML(
                """
                <div class="hero-kicker">A small computer-vision lab</div>
                <h1 class="hero-title">Make an image feel<br><em>like somewhere else.</em></h1>
                <p class="hero-copy">Neural Canvas separates the structure of one image from the visual language of another—then optimizes a new image, one pixel at a time.</p>
                """
            )
        with gr.Column(scale=3):
            gr.HTML(
                """
                <div class="hero-aside">
                  <div class="section-kicker">Under the hood</div>
                  <p><strong>Frozen VGG-19</strong> features guide the render. The network stays fixed; only the generated pixels move.</p>
                </div>
                """
            )

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=5, elem_id="inputs-panel"):
            gr.HTML(
                """
                <div class="section-heading">
                  <div><div class="section-kicker">01 / Inputs</div><div class="section-title">Set the visual brief</div></div>
                  <div class="section-note">two images · one result</div>
                </div>
                """
            )
            with gr.Row():
                with gr.Column():
                    gr.HTML('<p class="image-label"><span>01</span>Content image</p>')
                    content_input = gr.Image(
                        type="pil",
                        sources=["upload", "clipboard"],
                        show_label=False,
                        buttons=[],
                        height=211,
                        elem_classes=["upload-box"],
                    )
                    gr.HTML('<p class="image-help">The composition and shapes to keep.</p>')
                with gr.Column():
                    gr.HTML('<p class="image-label"><span>02</span>Style reference</p>')
                    style_input = gr.Image(
                        type="pil",
                        sources=["upload", "clipboard"],
                        show_label=False,
                        buttons=[],
                        height=211,
                        elem_classes=["upload-box"],
                    )
                    gr.HTML('<p class="image-help">The colors, texture, and visual rhythm to borrow.</p>')

            gr.Examples(
                examples=[["examples/content.png", "examples/style.png"]],
                inputs=[content_input, style_input],
                label="Try the included showcase pair",
                elem_id="examples-wrap",
                cache_examples=False,
            )

            with gr.Column(elem_classes=["controls"]):
                gr.HTML('<div class="section-kicker">02 / Recipe</div>')
                with gr.Row():
                    steps_input = gr.Slider(
                        50,
                        600,
                        value=250,
                        step=25,
                        label="Optimization steps",
                        info="More steps = finer convergence",
                    )
                    size_input = gr.Slider(
                        128,
                        768,
                        value=384,
                        step=64,
                        label="Maximum image edge",
                        info="384px is a good demo default",
                    )
                strength_input = gr.Slider(
                    4.0,
                    6.5,
                    value=5.0,
                    step=0.1,
                    label="Style intensity",
                    info="How strongly the reference influences the result",
                )
                preserve_input = gr.Checkbox(
                    value=False,
                    label="Preserve content colors",
                    info="Keep the source image's chroma while borrowing style texture",
                )

            with gr.Row(elem_classes=["action-row"]):
                run_button = gr.Button("Create stylized image  →", variant="primary", elem_id="run-button")
                clear_button = gr.Button("Reset", variant="secondary", elem_id="clear-button")
            gr.HTML('<p class="run-hint">The first run may download VGG-19 weights. CPU renders are intentionally small enough to run locally.</p>')

        with gr.Column(scale=7, elem_id="result-panel"):
            gr.HTML(
                """
                <div class="output-head">
                  <div><div class="section-kicker">03 / Output</div><div class="section-title">Your new visual system</div></div>
                  <div class="live-pill"><span class="live-dot"></span> ready to render</div>
                </div>
                """
            )
            output_image = gr.Image(
                type="pil",
                show_label=False,
                buttons=["download", "fullscreen"],
                height=470,
                elem_classes=["output-box"],
                placeholder="Your render will appear here",
            )
            run_summary = gr.HTML(_empty_summary(), elem_classes=["run-summary"])

    gr.HTML(
        """
        <section id="method">
          <div class="section-kicker">The method</div>
          <div class="method-title">A transparent pipeline, not a magic button.</div>
          <div class="method-grid">
            <article class="method-card"><div class="method-number">01 / EXTRACT</div><h3>Read visual structure</h3><p>A frozen ImageNet VGG-19 encodes content activations and style statistics at multiple depths.</p></article>
            <article class="method-card"><div class="method-number">02 / OPTIMIZE</div><h3>Move the pixels</h3><p>Adam minimizes content, normalized Gram-matrix style, and total-variation losses together.</p></article>
            <article class="method-card"><div class="method-number">03 / EXPORT</div><h3>Keep the result</h3><p>Inspect the loss breakdown, compare the output at full size, and download a clean RGB image.</p></article>
          </div>
        </section>
        <footer id="site-footer"><span>Built with PyTorch + Gradio</span><span>Neural Canvas / 2026</span></footer>
        """
    )

    run_button.click(
        fn=stylize,
        inputs=[
            content_input,
            style_input,
            steps_input,
            size_input,
            strength_input,
            preserve_input,
        ],
        outputs=[output_image, run_summary],
    )

    clear_button.click(
        fn=_reset_workspace,
        outputs=[content_input, style_input, output_image, run_summary],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=gr.themes.Base(), css=APP_CSS)
