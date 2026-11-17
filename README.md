# EEG-Based Multi-Neurodevelopmental Disorder Detection

Detecting ADHD and intellectual and developmental disorder (IDD) from EEG with one common
pipeline, evaluated only on people the model has never seen. B.Tech project, IIT Indore.

**Pipeline:** SMVMD -> common modes (delta, theta, alpha, beta) -> channel covariance ->
Riemannian tangent space -> LightGBM -> majority vote per person.

## Key results (unseen subjects)
- ADHD (121 children): 71.3% balanced accuracy, 93/121 children correct
- IDD (14 participants): 14/14 correct in rest, music and rest+music
- About 56 ms per 5-s window on one CPU core; no GPU needed
- The same pipeline scores 92.7% on ADHD if a person's windows are split between training
  and test, but 71.3% on new children: window-level evaluation inflates accuracy.

## Data (not included)
- DS-1 (IDD): Sareen et al., Data in Brief 2020 (Mendeley Data). Place in `idd/Data/`.
- DS-2 (ADHD): Motie Nasrabadi et al., IEEE DataPort 2020, doi:10.21227/rzfh-zn36.
  Place the zip files in `adhd/`.

## Reproduce
    pip install -r requirements.txt
    # commands in order: see Appendix B of report/report2/BTP_report2.tex
    python report/build_final.py
    python report/build_report2.py

All result files are in `results/`; the report numbers are generated from them.

> Research project, not a diagnostic tool. In the ADHD dataset the ADHD children were on
> methylphenidate and the controls were not, which limits the interpretation of ADHD results.
