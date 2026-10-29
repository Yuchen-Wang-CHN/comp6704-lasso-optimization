# Third-party materials

The project-specific solver implementations, experiment scripts, tests, and
documentation are distributed under the root MIT License.

## Figure quality checks

The following utilities are copied from
[Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills),
commit `4a04b953c13c57adc51e202ce296597f00d29d67`:

- `tools/audit_panel_alignment.py`
- `tools/audit_pdf_text.py`
- `tools/audit_figure_collisions.py`
- `tools/validate_figure.py`

These files retain their upstream Apache License 2.0. The complete license is
included in `LICENSES/nature-skills-Apache-2.0.txt`. Source URLs and checksums
are recorded in `tools/vendor_provenance.json`. The files are distributed
without local modifications, with their upstream notices retained.

## Online News Popularity dataset

Source: Fernandes, Vinagre, Cortez, and Sernadela (2015), *Online News
Popularity*, UCI Machine Learning Repository,
[DOI: 10.24432/C5NS3V](https://doi.org/10.24432/C5NS3V).
The dataset is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

The runner downloads the original dataset from UCI. The checked-in provenance
manifest records its source, hashes, schema, and preprocessing. News-derived
outputs use all 39,644 rows, 58 predictors, and `log1p(shares)`, with training-only
standardization and the documented random splits.

## Report style

The report uses the official NeurIPS 2025 LaTeX style. Run
`python tools/fetch_report_style.py` to retrieve the style directly from
[the official archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip).
Its upstream author notices are retained in the downloaded file.
