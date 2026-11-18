# Group 04 Final Project — NFL Draft

## Research question

How accurately can draft position and player characteristics distinguish late-round NFL Draft selections with at least one recorded Pro Bowl selection from those with none?

## Original dataset description

The NFL Draft 1985–2015 dataset, published by Ulrik Thyge Pedersen and downloaded from Kaggle [1], contains 8,435 draft-selection records and 34 variables covering draft details, player characteristics, and NFL performance. Among the 7,190 records with age reported, mean recorded age was 22.64 years (SD = 0.84), median age was 23 years (interquartile range: 22–23), and ages ranged from 20 to 29 years. Of records with known age, 7.1% were aged 20–21, 81.2% were aged 22–23, and 11.8% were aged 24–29; age was missing for 1,245 records (14.8% of the full dataset). The dataset includes 15 standardized position categories, with defensive backs (18.6%), linebackers (13.3%), and wide receivers (12.4%) the most common. College information is available for 5,005 records (59.3%), representing 317 distinct recorded college labels. Sex, race, and ethnicity are not recorded and cannot be summarized from this file.

These full-source statistics were recorded before reservation. Position and college describe the player sample; they are not demographic attributes. Age percentages use the 7,190 nonmissing ages; position and college-availability percentages use all 8,435 records. Percentages may not sum to 100 because of rounding. There are 3 identifying fields and 31 other variables, including outcomes; 34 is the raw variable count, not the number of model inputs.

## Repository

| GitHub Repository | Link |
| --- | --- |
| Group 04 Final Project | [https://github.com/V1elendank/group-04-final-project](https://github.com/V1elendank/group-04-final-project) |

This repository will contain the project's final results and report. The required `Data` folder already contains the frozen 20% reporting set. At 190,839 bytes (about 191 KB), it does not require Google Drive hosting.

| Partition | Records | Share | Use |
| --- | ---: | ---: | --- |
| Development | 6,748 | 80% | Exploration, training, and internal validation |
| Reporting | 1,687 | 20% | Final evaluation only |

## Run the notebook

Install dependencies with `python -m pip install -r requirements.txt`, then open `nfl_dataset_description.ipynb` in Jupyter or VS Code and select Run All. The notebook displays the saved original description and computes additional summaries **only on development data**. It never reads the reporting file. Python 3.13 was used for verification.

```bash
python scripts/verify_reporting_integrity.py
```

`scripts/create_split.py` documents the one-time reservation and refuses to overwrite existing files. Do not rerun it during modeling. The original full CSV is deliberately not copied into the repository.

## Classification and evaluation plan

- Target: `success = (pb >= 1)`. A zero means no Pro Bowl is **recorded in this snapshot**, not proof of an unsuccessful completed career. Pro Bowl recognition is one operational definition of talent.
- Late rounds: 4–7; early rounds: 1–3. Predeclare a primary modeling cohort of **1994–2010**, rounds 1–7. This keeps the seven-round era and gives each class at least six completed calendar seasons through 2015. It reduces, but does not eliminate, unequal follow-up; awards are not dated and complete lifetime outcomes are unavailable. The full source is reserved 80/20 first; apply this same fixed eligibility rule to each partition. Do not exclude ineligible reporting rows from the frozen file or inspect their counts now.
- Train on eligible development records from rounds 1–7; assess late-round performance as the primary question and early-round performance as a secondary comparison at the final phase. This evaluates unseen selections from the historical population, **not future draft classes**.
- Predictors available at the time of selection: overall pick, age, standardized position, college, and optionally drafting team. Draft year can provide era context. Use round as an alternative representation or reporting subgroup; compare against a pick-only baseline. Do not use player identifiers, `pb`, `ap1`, `hof`, `to`, `g`, `st`, any AV measure, or career performance statistics as predictors.
- Compare a majority-class baseline, a pick-only logistic regression, multivariable logistic regression, and a random forest. These are planned models; none has been fitted yet.
- Perform five-fold stratified **grouped** cross-validation within development data only. Keep nonmissing player IDs together. Fit imputation, categorical encoding, scaling, feature selection, and any class balancing inside each training fold using a pipeline. Use validation folds to choose model settings and a decision threshold.
- Primary ranking measure: average precision (a precision–recall summary). Also report precision, recall, F1, balanced accuracy, confusion matrices, and calibration. Do not optimize ordinary accuracy alone. Freeze all choices before evaluating the reporting set once; report uncertainty, especially for the smaller late-round subgroup.

## Why this split is random

Use a **stratified random split with player grouping**, not an unstratified row shuffle. The positive class is uncommon, so stratification maintains approximately comparable class proportions; grouping prevents the two selections for a known repeat-drafted player from appearing on both sides. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` reserves the first fold and uses the other four for development. With the recorded library versions, this gives exactly 6,748 development rows and 1,687 reporting rows. No seed or fold was chosen based on model performance.

This choice matches retrospective classification across comparable historical records. It does not establish accuracy on future draft classes. For a future-draft forecasting claim, use chronological validation with a consistently observed outcome window. Random splitting also does not fix the unequal career follow-up in this dataset. Earlier advice to use a chronological split applied to a future-class forecasting design; the present design explicitly makes a historical classification claim. See the scikit-learn [cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

## Reporting-set policy

`Data/reporting_set.csv` is reserved for final reporting. Do not open it for EDA, display its rows, summarize its outcomes, use it for preprocessing or feature selection, or tune models/thresholds against it. The working notebook reads only `Data/development_set.csv`. Do not regenerate or overwrite the split.

The manifest records row counts, the algorithm, seed, software versions, and SHA-256 checksums. Integrity checks only hash bytes; they do not parse records or evaluate predictions. Checksums detect changes, but do not prevent a person from opening a public GitHub file. Team members must follow this policy.

**Prior exposure:** full-dataset demographics and some aggregate outcome summaries were inspected before this reservation. No model was trained or tuned. The reporting file is untouched for analysis **from reservation onward**, not a dataset that has never previously contributed to any summary. Disclose this distinction to the instructor if their rule requires a completely unexamined sample.

## Reference

[1] U. T. Pedersen, “NFL Draft 1985–2015,” Kaggle, data set. Accessed: Nov. 17, 2026. [Online]. Available: https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015
