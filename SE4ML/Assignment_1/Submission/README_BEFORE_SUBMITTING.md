# Before uploading to Taxila

This assignment requires **two** deliverables (per the Assignment_1.pdf General Instructions and Submission Guidelines):

1. **`Group216.docx`** — the report (requirements formulation, GR4ML views, architecture, screenshots), plus **Appendix A (full unabridged source of `train_loan_model.py` and `app.py`)** and **Appendix B (the `train_output.txt` training log)** at the end, per the Submission Guidelines' requirement to include code inside the document. The docx is now 24 pages.
2. **`216.ipynb`** — the implementation notebook (naming convention `<Group no>.ipynb`). Already executed end-to-end against `data/loan_data.csv`; reproduces the Pipe-and-Filter training pipeline and the CQRS query/prediction step with real outputs (metrics, confusion matrix, sample predictions) matching the report.

## Steps

1. Open `Group216.docx` in Word and fill in:
   - Submission Date (title page)
   - Group Member Details table: **Qualitative Contribution is already pre-filled** for all 4 members (mapped to the 4 real work-streams of this project: requirements/GR4ML modeling, data-prep pipeline, model training/evaluation, architecture/app/report). You still need to fill in **BITS ID, Name, and % Contribution** — appears twice: title page and Section 7.
2. Open `216.ipynb` and fill in the same Group Member Details table in the first (markdown) cell — Contribution is pre-filled the same way; BITS ID/Name/% still placeholders.
3. Both files already mention Group No: 216 and the required name/group details section, per the PDF's instruction "Inside each report and implementation notebooks, you are required to mention your name, Group details."
4. Delete this README before zipping/uploading — it is not part of the deliverable.
5. Upload `Group216.docx` (or rename to match your exact required naming, e.g. `216.docx`) and `216.ipynb` to Taxila. The `code/`, `data/`, and `screenshots/` folders are kept here for your own records only.

Note: `216.ipynb` needs `data/loan_data.csv` in a `data/` subfolder next to it to re-run from scratch (already included here). If you upload the notebook standalone, keep `data/loan_data.csv` alongside it, or re-run isn't required since outputs are already baked in.
