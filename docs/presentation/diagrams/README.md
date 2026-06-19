# TikZ diagrams

Edit the `.tex` sources here, then render them into `docs/figures`.

For pure TikZ diagrams, use SVG:

```bash
latex -interaction=nonstopmode -halt-on-error ns_problem.tex
dvisvgm ns_problem.dvi -o ../../figures/ns/ns_problem.svg
```

For diagrams that include PNG fields, use LuaLaTeX to PDF, then rasterize:

```bash
lualatex -interaction=nonstopmode -halt-on-error ns_mean_correction.tex
pdftoppm -png -singlefile -r 220 ns_mean_correction.pdf ../../figures/ns/ns_mean_correction_diagram
```

The generated assets are included from `slides.md`, so VS Code's Marp preview
updates after the asset is regenerated.
