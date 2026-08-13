# Demo assets

The two inputs are generated deterministically by code in this repository, so they can be redistributed safely:

```bash
python examples/generate_demo_inputs.py
```

The checked-in result was produced by the actual pretrained VGG-19 path:

```bash
nst examples/content.png examples/style.png \
  --output examples/result.png \
  --size 256 \
  --steps 180 \
  --style-weight 100000 \
  --device cpu
```

For a personal portfolio, replace or supplement these synthetic examples with original photography and artwork that you have permission to publish.
