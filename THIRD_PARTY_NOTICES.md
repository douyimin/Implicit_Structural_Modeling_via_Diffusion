# Third-party notices

The project's existing MIT license applies to original contributions. Derived
files retain upstream terms; the project license does not replace them.

| Upstream | Derived portions | License |
| --- | --- | --- |
| [ControlNet](https://github.com/lllyasviel/ControlNet) | Conditional control model, model-loading/runtime and sampling adaptations | [Apache-2.0](licenses/ControlNet.txt) |
| [Latent Diffusion](https://github.com/CompVis/latent-diffusion) | VAE, diffusion and distribution utilities | [MIT](licenses/latent-diffusion.txt) |
| [guided-diffusion](https://github.com/openai/guided-diffusion) | UNet and diffusion utility components | [MIT](licenses/guided-diffusion.txt) |

Stable Diffusion 2.1 model weights have their own model-license conditions,
including the CreativeML Open RAIL++-M terms associated with the selected
upstream checkpoint. This source release contains no pretrained model weights.
Consult the exact checkpoint's distribution and license when obtaining weights.

The inherited attention implementation cites Hugging Face diffusers; the
corresponding attribution links remain in the source. OpenCLIP and Transformers
are dependencies installed separately, with their own licenses.

Release modifications include separate fault/horizon conditioning, geological
data handling, CLI/configuration, EMA coverage, native PyTorch attention and
checkpointing, device portability, and packaging. This notice records project
modifications without asserting an unverified original upstream commit.

## Figures

The three JPEG figures in `assets/` are reproduced from the associated manuscript
at the authors' request. The code's MIT license is not a blanket license for
third-party seismic or interpretation panels reproduced inside those figures.

- `fig_diffusion_framework.jpg`: framework figure from the associated manuscript.
- `fig_fold_thrust_examples.jpg`: includes source sections based on Butler, R.
  and Bond, C. (2020), *Thrust systems and contractional tectonics*, in
  *Regional Geology and Tectonics*, pp. 149–167, Elsevier.
- `fig_flower_structure_examples.jpg`: includes source sections and
  interpretations from Huang, L. and Liu, C.-Y. (2017), *Three types of flower
  structures in a divergent-wrench fault zone*, JGR: Solid Earth, 122(12).

Dataset terms are available with the [Zenodo archive](https://doi.org/10.5281/zenodo.18447817).
