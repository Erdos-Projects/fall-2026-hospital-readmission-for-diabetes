# fall-2026-hospital-readmission-for-diabetes

## Team Details
Team project: fall-2026-hospital-readmission-for-diabetes

Kevin Dao and Taylor Tan  

Erdős Institute Data Science Bootcamp, Fall 2026

**Question:** Can a small, interpretable model predict 30-day hospital readmission for patients with diabetes as well as a naive logistic regression using all available predictors and minimal preprocessing? 

## Data and outcome

We use the [Diabetes 130-US Hospitals for Years 1999–2008 dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), which contains 101,766 inpatient encounters. A patient may appear in more than one encounter. We label a recorded readmission within 30 days as positive and a later or unrecorded readmission as negative. In this dataset, 11.16% of encounters have the positive label.

Cleaning removes the encounter identifier, treats `?` and `NULL` as missing, and maps coded admission and discharge fields to descriptions. The patient identifier is retained to create grouped splits, then excluded from model inputs. Missing predictor values are handled by each model's preprocessing pipeline.

## Evaluation

We use one fold from a five-fold `StratifiedGroupKFold` to divide the encounters into 81,412 development and 20,354 test encounters. We divide the development set again into 65,129 training and 16,283 validation encounters. Encounters from the same patient stay on one side of each split, while the readmission rates remain approximately the same.

We fit candidate configurations on the training subset and select a classification threshold for each model by maximizing validation F2. We then refit each model on the full development set and apply its fixed validation-selected threshold to the test set. For precision $P$ and recall $R$, $F_2 = \frac{5PR}{4P+R}$. We report precision and recall alongside $F_2$ so the tradeoff between missed readmissions and false alerts remains visible.

## Results

| Model | Source inputs | Validation F2 | Fixed threshold | Test F2 | Test precision | Test recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Baseline logistic regression](results/final/baseline_model.ipynb) | 47 | 0.40878 | 0.06736 | 0.41277 | 13.4% | 85.9% |
| [Light logistic regression](results/final/logistic_regression_light.ipynb) | 7 | 0.42735 | 0.65989 | **0.42477** | 14.7% | 80.8% |


The light model had higher observed test $F_2$. Compared with the baseline at their fixed thresholds, it made 1,917 fewer false-positive predictions and identified 116 fewer actual readmissions in the 20,354 test encounters. 

## The seven-input model

The light model uses `number_inpatient`, `number_emergency`, `payer_code`, `discharge_disposition_id`, `diag_1`, `number_diagnoses`, and `num_medications`. Primary diagnosis codes in `diag_1` are grouped into broader categories before the seven inputs are selected. Numerical inputs receive median imputation and standard scaling. Categorical inputs receive a constant `Missing` category and one-hot encoding, with categories occurring fewer than 100 times grouped as infrequent. The fitted logistic regression uses `C=0.2` and class weights `{0: 1, 1: 22}`. Seven source inputs yield more than seven fitted coefficients after categorical encoding.

The saved [`final_model.joblib`](results/final/final_model.joblib) contains the fitted estimator pipeline for these selected inputs. 

**Warning:** Diagnosis grouping and input selection happen **before** that pipeline in the notebook. Apply those same steps before passing new encounters to the saved estimator. Its probability outputs have not been calibrated for clinical use.


## Repository map

| Path | Contents |
| --- | --- |
| `data/` | Original CSV files and prepared development and test splits in `cleaned_data/` |
| `models/modeling_data/` | Prepared training and validation splits |
| `models/preprocessing.py` and `models/scoring.py` | Shared data and scoring helpers |
| `models/` | Baseline experiments, comparisons with other models, and tuning of models |
| `notebooks/` | Exploratory analysis and feature-selection notebooks |
| `results/final/` | Final baseline and light test notebooks, plus the exported light estimator |


## Limitations and sources

The encounters were recorded from 1999 to 2008. These results do not establish performance at other hospitals or on current patients. Before clinical use, the model would require further external evaluation, probability calibration, checks across patient groups, and an assessment of how many follow-up alerts care teams can act on.

- [UCI dataset description](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- Strack, B., et al. (2014). [Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records](https://doi.org/10.1155/2014/781670). *BioMed Research International*, 2014, 781670.
