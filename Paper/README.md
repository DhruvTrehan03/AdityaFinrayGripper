# Generic Paper Template

This is a venue-neutral LaTeX paper scaffold. It uses a simple `article` layout so it can be adapted later to IEEE, ACM, Elsevier, Springer, MDPI, arXiv, or an institutional template once a venue is chosen.

## Build

```bash
make
```

The compiled PDF is written to `build/main.pdf`.

For continuous compilation while editing:

```bash
make watch
```

## Structure

- `main.tex`: preamble, title block, abstract, and section includes.
- `sections/`: paper body split by section.
- `refs.bib`: bibliography entries.
- `figures/`: place exported figures here.

## Venue Adaptation Notes

When you choose a venue, replace the document class and bibliography style first, then adjust margins, title block, author metadata, and section order to match the venue guidelines.
