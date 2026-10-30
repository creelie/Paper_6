# Homotopy Groups of Spheres, Hopf Fibrations and Villarceau Circles II

LaTeX source and verification code for the monograph of this title by Deep Bhattacharjee, version 2 of the preprint doi:10.20944/preprints202602.2038.v1.

## Layout

- `paper/`: the LaTeX source (`main.tex` and one file per chapter), the figures in `paper/figures/`, and the compiled `paper/main.pdf`. `paper/merge.py` writes `paper/manuscript.tex`, a single file with every chapter inlined, for submission.
- `code/checks.py`: 54 independent numerical and symbolic checks of the identities and values in the text (Appendix C). Run `cd code && python3 checks.py`; it needs NumPy and SymPy, takes about twenty seconds and ends with `54 checks passed, 0 failed`.
- `figures/make_figures.py`: produces the eight figures with Matplotlib.

## Build

```
cd paper && latexmk -pdf main.tex
```

The build has 0 errors, 0 overfull or underfull boxes and no undefined references with TeX Live 2026.

## Contact

Deep Bhattacharjee, d.bhattacharjee@erl-forschung.de, itsdeep@live.com; for GitHub-related contact egspl@proton.me. ORCID 0000-0003-0466-750X.

## License

MIT, see `LICENSE`.
