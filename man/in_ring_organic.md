# Organic Ink Ring Generator (Python)

This module generates **procedural ink rings** inspired by calligraphic and organic mark-making, similar in spirit to circular logograms and ink-based glyphs.

The generator produces:

* a **mostly linear circular ring**
* optional **gaps** (open vs closed forms)
* **articulated ink traces** (not simple blobs)
* subtle **spray and texture**
* fully **seeded** output for reproducibility

It is designed for generative art, visualization, and symbolic geometry.

---

## Conceptual model

The ring is constructed as:

1. A **base circle** with controlled smoothness
2. Optional **gaps** that break the continuity of the ring
3. **Organic ink traces**:

   * short random-walk strokes
   * variable thickness
   * directional bias (outward / inward)
4. Optional **micro-spray** for texture

All components are driven by interpretable parameters.

---

## Function definition

```python
import numpy as np
import matplotlib.pyplot as plt


def ink_ring_organic(
    seed=0,
    radius=1.0,
    linearity=0.92,     # 0..1 (higher = smoother ring)
    ink_gap=0.08,       # 0..1 (higher = more open ring)
    streaks=0.35,       # 0..1 (articulated ink traces)
    complexity=0.12,    # 0..1 (micro spray / roughness)
    figsize=(6, 6),
    show=True,
    save=None,
    bg="white",
):
    """
    Generate an organic ink ring with articulated ink traces.

    Parameters
    ----------
    seed : int
        Random seed controlling reproducibility.
    radius : float
        Base radius of the ring.
    linearity : float
        Controls smoothness of the ring (1 = very clean).
    ink_gap : float
        Controls openness of the ring (0 = closed).
    streaks : float
        Amount of articulated ink traces.
    complexity : float
        Fine-scale noise and spray.
    figsize : tuple
        Figure size.
    show : bool
        Display the plot.
    save : str or None
        Save path for PNG output.
    bg : str
        Background color.

    Returns
    -------
    fig, ax : matplotlib objects
    """
    ...
```

*(Function body omitted here for brevity; see source code.)*

---

## Parameters overview

| Parameter    | Range       | Effect                                |
| ------------ | ----------- | ------------------------------------- |
| `seed`       | int         | Reproducible glyph variants           |
| `radius`     | float       | Size of the ring                      |
| `linearity`  | 0–1         | Higher = cleaner, more uniform ring   |
| `ink_gap`    | 0–1         | Higher = more open ring (larger gaps) |
| `streaks`    | 0–1         | Number and complexity of ink traces   |
| `complexity` | 0–1         | Micro spray and roughness             |
| `save`       | path / None | Save PNG output                       |
| `show`       | bool        | Display result                        |

---

## Usage examples

### Clean, almost closed ring

```python
ink_ring_organic(
    seed=7,
    linearity=0.97,
    ink_gap=0.03,
    streaks=0.20,
    complexity=0.05
)
```

### Balanced ring with expressive traces

```python
ink_ring_organic(
    seed=7,
    linearity=0.92,
    ink_gap=0.06,
    streaks=0.35,
    complexity=0.10
)
```

### More expressive, chaotic ink

```python
ink_ring_organic(
    seed=7,
    linearity=0.80,
    ink_gap=0.15,
    streaks=0.70,
    complexity=0.30
)
```

---

## Design principles

* **Seeded**: every glyph is reproducible
* **Parametric**: each visual aspect is independently controllable
* **Procedural**: no images or textures, only geometry
* **Original**: inspired by calligraphy and ink behavior, not copying any specific symbol set

---

## Intended use cases

* Generative art
* Symbolic or fictional writing systems
* Visual metaphors for time, cycles, or processes
* Research and teaching of procedural geometry
* Inclusion in a nature/geometry Python package

---

## Possible extensions

* Directional “heavy sector” (one dominant ink region)
* SVG export
* Animated ink growth
* Grid / catalogue generation
* Integration into a package such as
  **`geomorphPy` / `biomorphPy` / `naturaform`**

---

## Next steps:

* write a **package-level README**
* design a **consistent visual style** across ferns, shells, rings, snowflakes

