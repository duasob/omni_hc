# OmniHC Presentation

This folder contains a Marp presentation authored in Markdown and intended to
be edited with VS Code.

## Edit and preview

Open this folder as the VS Code workspace:

```bash
code docs/presentation
```

Install the recommended **Marp for VS Code** extension when prompted. Open
`slides.md`, then use VS Code's normal Markdown preview:

- `Cmd+Shift+V` opens the preview.
- `Cmd+K`, then `V` opens the preview to the side.

There is no separate `Marp: Open Preview` command. The extension replaces the
normal Markdown preview with the slide preview when `marp: true` is present in
the document front matter.

Slides are separated with `---`. Presentation media lives in `docs/figures/`.
The local `assets/` path in this folder is a symlink to that source directory,
so slides should use domain-scoped relative paths:

```markdown
![Model rollout](assets/ns/ns_problem_rollout.gif)
```

Use the browser version when presenting so animated GIFs continue to play.

## Deck switches

The deck uses the peril-inspired palette from `theme.css`. Change the `class:`
line in the `slides.md` front matter to switch theme:

```yaml
class: light
```

Use `dark` instead of `light` for the dark theme. To hide the bottom
Problem / Constraint / Results progress pills, set the top-level CSS variable
in `slides.md` to `0`:

```yaml
style: |
  section {
    --slide-state-progress-enabled: 0;
  }
```

To override a single slide theme, put this immediately after its opening `---`:

```markdown
<!-- _class: dark -->
```

Use `<!-- _class: light -->` for the inverse.

## Export

Run `Marp: Export Slide Deck` from the VS Code command palette. Export HTML for
the animated presentation and PDF as a static backup. PDF pages cannot preserve
GIF animation.

The repository-level VS Code settings load `theme.css` and enable the small
HTML blocks used by the starter slides. No npm setup is required for editing.

## Build shareable HTML

Use the build script to refresh the `assets/` symlink, copy the referenced
figure files into `dist/assets`, export HTML, and create a zip that can be
shared without depending on repository paths:

```bash
docs/presentation/build.sh
```

The script uses `docs/presentation/node_modules/.bin/marp` when present, or a
`marp` binary on `PATH`. To use a custom binary, set `MARP_BIN`:

```bash
MARP_BIN=/path/to/marp docs/presentation/build.sh
```

PDF export also needs `--allow-local-files`, but animated assets will render as
static frames.
