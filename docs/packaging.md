# Packaging

How the `pesto` command is packaged, so that dtand's target preparation can install it
with `uv tool install git+https://github.com/lkndl/PeSTo@<tag>` on Linux, macOS and
Windows, and what to keep in mind when changing it.

## Weights

- The package holds the four pre-trained models in `src/pesto/weights/<run>/`: the
  `model_ckpt.pt` with the `config.py` and `model.py` it was trained with. This is the only
  copy. The rest of each training run (`main.py`, `data_handler.py`, `model.pt`, logs) stays
  in `model/save/<run>/`, so the notebooks that build a `Dataset` put both folders on
  `sys.path`, and `md_analysis/save` links to `src/pesto/weights`.
- They are real files, not a link: a checkout without symlink support turns a link into a
  small text file, and the wheel then has no weights.
- `predict.py` loads `config.py` and `model.py` by path with `importlib`, under a module
  name of their own, and never puts the weights directory on `sys.path`.

## Version

- The version comes from git through setuptools-scm: a tag `v0.2.0` gives `0.2.0`, a later
  commit `0.2.1.devN+g<hash>`. It ends up in `provenance.tool_version` of every track.
- Never move a tag once pushed. uv caches a build per commit, so an installation from a
  moved tag can keep the version of the old one.

## Dependencies

- Every dependency needs a lower bound: `uv tool install` resolves from `pyproject.toml`
  and does not read `uv.lock`.
- On Linux, torch comes from the explicit `pytorch-cpu` index (`[tool.uv.sources]`), which
  saves the several GB of CUDA libraries that come with torch from PyPI. uv honours this
  for a git install too: checked with uv 0.7.3 and 0.13.7, `pesto @ git+…` resolves to
  `torch==…+cpu` on `x86_64` and `aarch64` Linux and to the PyPI build on Windows.
- The notebooks, the training and the profiling script need the `notebooks` dependency
  group, which a plain `uv sync` leaves out.

## Repository

- `MANIFEST.in` keeps the sdist to the package. Without it, setuptools-scm puts every file
  tracked by git into the sdist, about 900 MB of examples and data.
- `CLoNe` must not be a submodule without a URL in `.gitmodules`: uv runs
  `git submodule update --init` on every git install, which then fails.
