# 💉 RMDS Hackathon · Dataset B
### Who will accept an H1N1 vaccine, and who should the town send one to?

> A cost-sensitive machine-learning solution that predicts, person by person, who will accept an H1N1 vaccine, and sends a vaccine only when the expected hospital saving outweighs the risk of wasting \$500.

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Our Approach](#-our-approach)
3. [Repository Structure](#-repository-structure)
4. [Dataset](#-dataset)
5. [Data Exploration](#-data-exploration)
6. [Pre-processing](#-pre-processing)
7. [Cost Model](#-cost-model)
8. [Modelling](#-modelling)
9. [Performance Analysis](#-performance-analysis)
10. [Submission Files](#-submission-files)
11. [Explainable AI](#-explainable-ai)
12. [Final Answer](#-final-answer)
13. [Assignment Compliance Checklist](#-assignment-compliance-checklist)
14. [How to Run](#-how-to-run)
15. [Summary, Limitations & References](#-summary)

---

## 🎯 Problem Statement

Every town has many winter indoor parties, so **everyone in town will catch the flu** (100% infection probability). The town's goal is to **minimise the total money spent**, which has two parts:

| Cost | Value |
|---|---|
| 💉 H1N1 vaccine | **\$500** |
| 🏥 Hospitalisation for H1N1 | **\$25,000** |

**Rules of the game**
- A local law says everyone who wants a vaccine must get one. If a vaccine is not available, the town **pays the person \$500 compensation** instead.
- Vaccines are posted to people's homes and are **wasted if not used**, so we must know *exactly who* will accept, not just the total number.
- Hospitalisation probability depends on factors in the dataset and on vaccination status.

**Task:** predict the target variable `h1n1_vaccine` (1 = accepts the vaccine) for every person in the test set in the way that makes the town's total bill as small as possible.

---

## 🧠 Our Approach

> Train several models to estimate each person's chance of accepting, then send a vaccine **only when the expected hospital saving outweighs the risk of wasting \$500.**

| Section | What happens |
|---|---|
| 1. Setup | Libraries, settings, loading the data |
| 2. Data exploration | What the data looks like and what relates to vaccination |
| 3. Pre-processing | Handling missing answers and text columns |
| 4. Cost model | Turning a prediction into dollars |
| 5. Modelling | Eight candidate models, cross-validated |
| 6. Performance analysis | Standard metrics, then cost |
| 7. Submission files | The five best models |
| 8. Explainable AI | What drives the predictions |
| 9. Final answer | Which model saves the most money |
| 10. Summary and references | Key points, limitations, sources |

---

## 📂 Repository Structure

```
.
├── final_hackathon_solution.ipynb          # Full solution notebook (Google Colab)
├── README.md                               # This file
├── challenge_submission_group_0_order_1.csv   # Gradient boosting
├── challenge_submission_group_0_order_2.csv   # Extra trees
├── challenge_submission_group_0_order_3.csv   # Linear discriminant analysis
├── challenge_submission_group_0_order_4.csv   # Logistic regression
└── challenge_submission_group_0_order_5.csv   # Random forest
```

The dataset files (`dataset_B_training.csv`, `dataset_B_testing.csv`) are provided by the hackathon and are not stored in this repository. The notebook also saves its charts as PNG files when run.

---

## 📊 Dataset

| File | People | Columns |
|---|---|---|
| `dataset_B_training.csv` | **4,756** | 31 (includes the target `h1n1_vaccine`) |
| `dataset_B_testing.csv` | **4,749** | 30 (no target, this is what we predict) |

The dataset has 29 predictors (`respondent_id` is only an identifier): **19 numeric** and **10 text** features.

| Group | Features |
|---|---|
| **Concern & knowledge** | `h1n1_concern`, `h1n1_knowledge` |
| **Behaviour** (binary) | `behavioral_antiviral_meds`, `behavioral_avoidance`, `behavioral_face_mask`, `behavioral_wash_hands`, `behavioral_large_gatherings`, `behavioral_outside_home`, `behavioral_touch_face` |
| **Medical / exposure** | `doctor_recc_h1n1`, `child_under_6_months`, `health_worker`, `health_insurance`, `number_chronic_med_conditions` |
| **Opinions** (1–5 scale) | `opinion_h1n1_vacc_effective`, `opinion_h1n1_risk`, `opinion_h1n1_sick_from_vacc` |
| **Demographics** | `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status`, `census_msa`, `household_adults`, `household_children`, `employment_sector` |

---

## 🔍 Data Exploration

### Target balance and missing answers
### Vaccination rate by answer
### Correlation with the target
### Key findings
- **Balanced enough.** About **39%** of respondents were vaccinated, so no re-sampling is needed.
- **Strongest signals.** A doctor's recommendation, believing the vaccine works and feeling at risk. Health workers are also far more likely to be vaccinated.
- **Missing answers.** `employment_sector` (about 50% missing) and `health_insurance` (about 40% missing) have many gaps. Dropping those rows would lose half the data, so we **keep "did not answer" as information** the models can use.

---

## 🧹 Pre-processing

The same steps are applied to training and test data through **one pipeline**, so nothing learned from the test set can leak into the models.

| Column type | Missing answers | Then |
|---|---|---|
| Numbers (binary, 1–5 scales, counts), **19 columns** | Filled with the **median**, plus a "was missing" flag | Rescaled (`StandardScaler`) |
| Text (age group, education, ...), **10 columns** | Become their own **"missing"** category | One-hot encoded |

---

## 💰 Cost Model

The hackathon is judged on money, so we first define exactly what each decision costs. Since everyone is infected, there are four cases per person:

| We send a vaccine | They want it | Vaccine money | Hospital risk |
|---|---|---|---|
| Yes | Yes | \$500 | Reduced by the vaccine |
| Yes | No | \$500 (vaccine ruined) | Full |
| No | Yes | \$500 (compensation) | Full |
| No | No | \$0 | Full |

### Hospitalisation probability assumptions

| Input | Value | Source |
|---|---|---|
| Average risk if infected | **0.45%** | CDC: about 61 million US cases and 274,000 hospitalisations in the 2009 pandemic |
| Vaccine effectiveness | **62%** (43% for 65+) | Borse et al., *Emerging Infectious Diseases*, 2013 |
| Chronic conditions | **4x, 9x, 20x** risk for 1, 2, 3+ conditions | Ward et al., *Emerging Infectious Diseases*, 2011 |
| Age 65+ | **2x** risk | CDC surveillance (direction); the size is our assumption |

The multipliers are scaled so that the *average* person in the data has the 0.45% risk.

### Decision rule

Sending a vaccine is worth it when the expected hospital saving beats the expected waste:

```
Send a vaccine if:   P(accept)  >  500 / (500 + 25000 × risk × effectiveness)
```

The cut-off is therefore **personal**: the higher someone's hospital risk, the less sure we need to be that they will accept.

### Risk and cut-off for each type of person

| Chronic conditions | Age 65+ | People | Hospital risk | Send if P(accept) is above |
|---|---|---:|---:|---:|
| 0 | No | 2,616 | 0.14% | 0.96 |
| 0 | Yes | 697 | 0.28% | 0.94 |
| 1 | No | 713 | 0.56% | 0.85 |
| 1 | Yes | 401 | 1.12% | 0.81 |
| 2 | No | 153 | 1.26% | 0.72 |
| 2 | Yes | 102 | 2.52% | 0.65 |
| 3+ | No | 49 | 2.80% | 0.53 |
| 3+ | Yes | 25 | 5.61% | 0.45 |

**Reading the table:** a healthy adult under 65 only gets a vaccine if we are about **96%** sure they will accept, because a wasted vaccine costs \$500 while the expected hospital saving is small. For someone with three or more chronic conditions, about **50%** is enough.

---

## 🤖 Modelling

We compare **eight models** from different families and submit the best five.

| Model | Idea | Key settings |
|---|---|---|
| Logistic regression | Weighted sum of the answers | `C=0.1` |
| Linear discriminant analysis | Separates the two groups with a straight boundary | defaults |
| Decision tree | A flowchart of yes/no questions | `max_depth=5`, `min_samples_leaf=50` |
| K-nearest neighbours | Looks at the most similar people | `n_neighbors=75`, distance-weighted |
| Random forest | Many trees voting | 500 trees, `min_samples_leaf=8` |
| Extra trees | Like a random forest, with more random splits | 500 trees, `min_samples_leaf=8` |
| Neural network | Two hidden layers (64, 32) | `alpha=2.0`, early stopping |
| Gradient boosting | Trees built one after another, each fixing earlier mistakes | `learning_rate=0.02`, 500 iterations, `l2_regularization=10` |

**How they are scored.** 5-fold stratified cross-validation: every person is predicted by a model that never saw them, which imitates the test set.

**Tuning.** Hyperparameters are fixed in the notebook (see the *Key settings* column above); the notebook notes that five of the models were tuned by grid search on cross-validated log loss, and the rest use sensible defaults.

**Calibration.** The cost rule needs honest probabilities (a predicted 0.9 should mean 90%). Each model is tried **with and without isotonic calibration** and keeps the cheaper version.

| Model | Cost, raw | Cost, calibrated | Version used |
|---|---:|---:|---|
| Logistic regression | \$1,436,049 | \$1,436,465 | raw |
| Linear discriminant analysis | \$1,435,795 | \$1,436,235 | raw |
| Decision tree | \$1,441,456 | \$1,439,141 | calibrated |
| K-nearest neighbours | \$1,448,821 | \$1,439,595 | calibrated |
| Random forest | \$1,447,115 | \$1,436,745 | calibrated |
| Extra trees | \$1,447,288 | \$1,433,087 | calibrated |
| Neural network | \$1,437,045 | \$1,436,936 | calibrated |
| Gradient boosting | \$1,431,164 | \$1,433,424 | raw |

---

## 📈 Performance Analysis

### 6.1 Standard metrics (5-fold CV, usual 0.5 cut-off)

| Model | AUC | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Average of all models | 0.856 | 0.786 | 0.769 | 0.653 |
| **Gradient boosting** | **0.854** | 0.787 | 0.760 | 0.671 |
| Random forest | 0.850 | 0.787 | 0.781 | 0.639 |
| Logistic regression | 0.849 | 0.788 | 0.761 | 0.673 |
| Linear discriminant analysis | 0.849 | 0.787 | 0.762 | 0.667 |
| Neural network | 0.847 | 0.782 | 0.764 | 0.646 |
| Extra trees | 0.847 | 0.781 | 0.766 | 0.640 |
| K-nearest neighbours | 0.834 | 0.771 | 0.759 | 0.613 |
| Decision tree | 0.829 | 0.769 | 0.724 | 0.666 |

All eight models rank people about equally well (AUC 0.83–0.85).

### 6.2 Total cost (the metric that matters)

Each model's probabilities are turned into decisions with the cost rule and priced on the training set (cross-validated).

| Strategy | Total cost | Cost per person | Vaccines sent |
|---|---:|---:|---:|
| 🥇 **Gradient boosting** | **\$1,431,164** | **\$300.92** | 434 |
| Extra trees | \$1,433,087 | \$301.32 | 438 |
| Average of all models | \$1,434,819 | \$301.69 | 360 |
| Linear discriminant analysis | \$1,435,795 | \$301.89 | 451 |
| Logistic regression | \$1,436,049 | \$301.94 | 385 |
| Random forest | \$1,436,745 | \$302.09 | 460 |
| Neural network | \$1,436,936 | \$302.13 | 420 |
| Decision tree | \$1,439,141 | \$302.59 | 305 |
| K-nearest neighbours | \$1,439,595 | \$302.69 | 378 |
| Average of all models, **0.5 cut-off** | \$1,563,877 | \$328.82 | 1,590 |

**Reference points**

| Reference | Total cost |
|---|---:|
| Perfect knowledge (unreachable best case) | \$1,343,233 |
| Send to nobody | \$1,472,050 |
| Send to everybody | \$2,784,233 |

> 💡 Using the same probabilities (average of all models), the usual 0.5 cut-off costs **\$129,058 more** than the cost rule (\$1,563,877 vs \$1,434,819) and sends 1,590 vaccines instead of 360. This is why the cost rule matters.

### 6.3 Where the money goes
Sending vaccines to everybody wastes far more than it saves.

### 6.4 Where the best model is right and wrong
The model sends few vaccines, but almost all of them are used. Many people who wanted a vaccine are not sent one, and that is **deliberate**: for low-risk people, paying compensation is cheaper than gambling on a wasted vaccine. Only the far right of the probability distribution clears the cut-off.

### 6.5 Sensitivity check

Our hospitalisation risk is an estimate. Does the best model still beat "send to nobody" if the true average risk is half, double or four times our figure?

| Average risk | Best model | Send to nobody | Saving | Vaccines sent |
|---|---:|---:|---:|---:|
| 0.22% | \$1,190,772 | \$1,204,525 | \$13,753 | 248 |
| **0.45% (our estimate)** | \$1,431,164 | \$1,472,050 | \$40,886 | 434 |
| 0.90% | \$1,901,782 | \$2,007,100 | \$105,318 | 692 |
| 1.80% | \$2,816,392 | \$3,077,200 | \$260,808 | 1,026 |

✅ **The model saves money in every scenario**, and the saving grows quickly as the risk rises, because more vaccines become worth sending.

---

## 📤 Submission Files

The **five cheapest single models** are written in the format of `dataset_B_example_submission.csv`, with two columns: `respondent_id` and `h1n1_vaccine` (1.0 = send a vaccine, 0.0 = do not). File names follow the rule `challenge_submission_group_X_order_Y.csv`, where order 1 is the best result.

| File | Model | Vaccines sent |
|---|---|---:|
| `challenge_submission_group_0_order_1.csv` | Gradient boosting | 429 of 4,749 |
| `challenge_submission_group_0_order_2.csv` | Extra trees | 425 of 4,749 |
| `challenge_submission_group_0_order_3.csv` | Linear discriminant analysis | 459 of 4,749 |
| `challenge_submission_group_0_order_4.csv` | Logistic regression | 392 of 4,749 |
| `challenge_submission_group_0_order_5.csv` | Random forest | 471 of 4,749 |

Example format:

```csv
respondent_id,h1n1_vaccine
1,0.0
2,1.0
...
```

---

## 🔬 Explainable AI

### 8.1 Which features matter most?
Permutation importance: shuffle one feature at a time and measure how much the best model's AUC drops.

### 8.2 In which direction does each feature push?
Logistic regression coefficients: blue bars make accepting the vaccine more likely, orange bars less likely.

### What drives the predictions
- 🩺 A **doctor's recommendation**
- 🏥 **Health insurance** (including whether the question was answered at all)
- 🧍 The person's own **opinion of H1N1 risk** and **vaccine effectiveness**

These match what the exploration in the data-exploration section suggested.

---

## 🏆 Final Answer

Every model is compared with the simplest alternative: sending no vaccines at all (\$1,472,050).

| Rank | Model | Total cost | Saving vs no vaccines | Saving % | Vaccines sent | Submitted as |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Gradient boosting** | \$1,431,164 | \$40,886 | 2.78% | 434 | order_1 |
| 2 | Extra trees | \$1,433,087 | \$38,963 | 2.65% | 438 | order_2 |
| 3 | Linear discriminant analysis | \$1,435,795 | \$36,255 | 2.46% | 451 | order_3 |
| 4 | Logistic regression | \$1,436,049 | \$36,001 | 2.45% | 385 | order_4 |
| 5 | Random forest | \$1,436,745 | \$35,305 | 2.40% | 460 | order_5 |
| 6 | Neural network | \$1,436,936 | \$35,114 | 2.39% | 420 | not submitted |
| 7 | Decision tree | \$1,439,141 | \$32,909 | 2.24% | 305 | not submitted |
| 8 | K-nearest neighbours | \$1,439,595 | \$32,455 | 2.20% | 378 | not submitted |

### 🥇 Best model: Gradient boosting
- **Total cost:** \$1,431,164 (**\$300.92 per person**)
- **Saves \$40,886 (2.78%)** compared with sending no vaccines

---

## ✅ Assignment Compliance Checklist

| Hackathon requirement | Status | Where |
|---|---|---|
| Data exploration | ✅ | Data Exploration |
| Pre-processing | ✅ | Pre-processing |
| Modelling (several model families) | ✅ 8 models | Modelling |
| Performance analysis | ✅ AUC, accuracy, precision, recall, **total cost**, sensitivity | Performance Analysis |
| Explainable AI | ✅ Permutation importance and logistic coefficients | Explainable AI |
| Code presentation and documentation | ✅ Notebook sections plus this README | whole repo |
| Submit the best **5** results | ✅ 5 files | Submission Files |
| File name `challenge_submission_group_X_order_Y` | ✅ `X` = `GROUP`, `Y` = rank (1 = best) | Submission Files |
| CSV with `respondent_id` and the predicted target | ✅ columns `respondent_id`, `h1n1_vaccine` | Submission Files |
| Costs: vaccine \$500, hospitalisation \$25,000 | ✅ `VACCINE_COST`, `HOSP_COST` | Cost Model |
| 100% infection, compensation when no vaccine, wasted unused vaccines | ✅ all three built into `total_cost` | Cost Model |
| Hospitalisation probability from dataset factors and vaccination status | ✅ age, chronic conditions, vaccine effectiveness | Cost Model |

---

## 🚀 How to Run

1. Open `final_hackathon_solution.ipynb` in **Google Colab**.
2. Set `GROUP` in the first code cell (this is used in the submission file names).
3. Choose *Runtime → Run all*. The notebook mounts Google Drive in its second code cell; approve the prompt (or skip it, since the data is loaded through the uploader).
4. When the **Choose Files** button appears, select **both** the training and testing CSV files. The file with `train` in its name is used for training and the one with `test` in its name for testing; any file with `submission` in its name is ignored.
5. The five submission CSVs and the notebook's charts (PNG files) are saved in the working folder.

**Requirements**
```
numpy
pandas
matplotlib
scikit-learn >= 1.2   # OneHotEncoder(sparse_output=...) needs 1.2+
ipython                # optional, for table display
```

Outside Colab, the file uploader falls back to the CSV files in the current folder. The Google Drive mount cell (`google.colab`) is Colab-only and should be skipped or removed when running locally.

**Reproducibility:** a fixed seed (`SEED = 42`) is used everywhere, so every run gives the same result.

---

## 📝 Summary

### What we did
- Compared **eight separate models** on the town's total cost and submitted the best five.
- Built a cost model from published H1N1 figures and used it to set a **personal cut-off for every respondent**.

### What we found
- A wasted vaccine costs \$500, while an accepted one saves far less than that in expected hospital costs for a healthy adult. So vaccines go **only to people very likely to accept**, with a lower bar for people with chronic conditions.
- The cost rule clearly beats the usual 0.5 cut-off and beats sending no vaccines.
- The top models are close together, so their exact order may differ on the test set.
- Main drivers: a doctor's recommendation, health insurance, and opinions on H1N1 risk and vaccine effectiveness.

### ⚠️ Limitations
- The hospitalisation risks come from published studies but are still estimates; the size of the age effect is our own assumption. The sensitivity check shows the conclusion holds across a wide range of risk levels.
- Costs are measured on the training data with cross-validation; the test set may differ.

### 📚 References
1. CDC. *Updated CDC Estimates of 2009 H1N1 Influenza Cases, Hospitalizations and Deaths in the United States, April 2009 – April 10, 2010.*
2. Borse RH, et al. *Effects of Vaccine Program against Pandemic Influenza A(H1N1) Virus, United States, 2009–2010.* Emerging Infectious Diseases, 2013.
3. Ward KA, Spokes PJ, McAnulty JM. *Case-Control Study of Risk Factors for Hospitalization Caused by Pandemic (H1N1) 2009.* Emerging Infectious Diseases, 2011.
4. CDC. *2009 H1N1 Early Outbreak and Disease Characteristics.*

---

<p align="center"><i>RMDS Hackathon · Data Science Challenge · Dataset B</i></p>
