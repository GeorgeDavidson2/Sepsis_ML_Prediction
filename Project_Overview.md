# Project Overview — Plain Language Guide
## Sepsis Early Prediction Using Machine Learning
### George Arthur & Promise Owa | CS6140

---

## What Is Sepsis?

Sepsis is a medical emergency. It happens when the body's response to an infection spirals out of control and starts attacking its own organs. It can kill within hours.

- **1.7 million** Americans develop sepsis every year
- **350,000** die from it — more than prostate cancer and breast cancer combined
- Every hour doctors delay treatment, the patient's chance of dying increases by **4 to 8%**

The good news: ICU patients are already connected to monitors that record their vital signs and run blood tests every hour. All the data needed to catch sepsis early already exists. The problem is that no one has reliably figured out how to turn that data into an early warning in time.

That is what this project is about.

---

## What Are We Trying To Do?

We are building a machine learning system that looks at a patient's hourly ICU data and predicts: **"This patient is going to develop sepsis in the next 6 hours."**

If a doctor gets that warning 6 hours early, they can start treatment before the patient crashes. That window can mean the difference between life and death.

---

## The Real Question We Are Asking

Many researchers have already tried to build sepsis prediction systems using machine learning. Some work better than others. But here is the problem with the existing research:

When researchers improve their system, they usually change **two things at the same time**:
1. How they clean and prepare the data (called **preprocessing**)
2. What type of AI model they use (called the **model architecture**)

Because they change both at once, nobody knows which change actually made the difference. Was it the better data preparation? Or the fancier model?

**Our project isolates these two things and tests them separately.**

We want to answer: which matters more — better data preparation, or a more sophisticated model?

---

## Our Experiment — The 2×2 Design

We designed a controlled experiment. Think of it like a science fair project where you only change one thing at a time.

We have **two data preparation strategies** and **two model types**. We cross them to get four combinations:

|  | Simple Data Prep | Smart Data Prep |
|---|---|---|
| **Simple Models** | Combination A | Combination B |
| **Advanced Model** | Combination C | Combination D |

By comparing the results of these four combinations, we can answer very specific questions:

- **A vs B** → Does better data preparation help, even with a simple model?
- **A vs C** → Does a fancier model help, even with basic data preparation?
- **B vs D** → Does the fancier model still add value when data prep is already excellent?
- **A vs D** → What is the total gain when we use both improvements together?

---

## The Data

We are using a publicly available dataset from the **PhysioNet/CinC 2019 Challenge** — a major international medical AI competition.

- **40,336 ICU patients** from two different hospitals
- Each patient has **hourly measurements** of 40 things: heart rate, blood pressure, temperature, blood test results, and more
- We know exactly when each patient developed sepsis (or didn't)
- We shift the labels 6 hours back so the task becomes: predict sepsis 6 hours before it happens

This is one of the most respected and widely used datasets in clinical AI research.

---

## The Two Data Preparation Strategies

### Strategy A — Simple (Median Fill)
ICU data has a lot of gaps. Blood tests like lactate or creatinine are not run every hour — only when a doctor suspects something is wrong. So most of those cells are empty.

The simple approach: fill every empty cell with the average value for that measurement across all patients.

**Problem:** This throws away important information. A missing lactate reading does not mean the patient is average — it might mean no one was worried enough to test yet, or it might mean the lab result hasn't come back. That context matters.

### Strategy B — Smart (Forward-Fill + Missingness Flags)
The smarter approach does two things:

1. **Forward-fill:** If a blood test was run 3 hours ago, use that value until a new one arrives. This preserves the last known clinical state rather than replacing it with a population average.

2. **Missingness flags:** Add a new column for each blood test that says: "Was this value freshly measured at this hour, or are we using an old value?" This gives the model a signal about what the doctors were watching closely — which is itself clinically informative.

**Our central hypothesis:** Strategy B will outperform Strategy A significantly, because *which values are missing* tells you something real about the patient's condition.

---

## The Two Model Types

### Simple Models — Logistic Regression and XGBoost
These models look at each hour of data as a snapshot in isolation. They ask: "Given everything we know about this patient right now, how likely is sepsis?"

- **Logistic Regression** is the simplest possible model. It is our baseline — the floor every other model should beat.
- **XGBoost** is a more powerful version that also looks at recent trends (e.g., heart rate 1 hour ago, 2 hours ago) to get a basic sense of whether things are getting better or worse.

### Advanced Model — LSTM (Long Short-Term Memory)
An LSTM is a type of neural network designed specifically to process sequences over time. Instead of looking at each hour as an isolated snapshot, it reads the patient's entire ICU timeline from admission to the current moment.

It can learn patterns like: "Heart rate slowly rising over 8 hours, combined with blood pressure gradually dropping and lactate trending upward" — the full temporal story of a patient deteriorating.

---

## Our Central Hypothesis

We predict that **better data preparation (Strategy B) will contribute more to performance than switching to the advanced LSTM model.**

Here is why we believe this:

- The pattern of *which tests are missing* is clinically meaningful — doctors order tests when they are worried. A missing lactate at hour 6 means something different from a missing lactate at hour 1.
- The LSTM is powerful, but the XGBoost model already has access to recent trends through its lag features. The gap between "smart tabular model with lag features" and "full sequence model" may be smaller than the gap between "naive imputation" and "missingness-aware imputation."

We designed the experiment to confirm or refute this. Either result is interesting.

---

## How We Measure Success

We do not use accuracy. Here is why: if only 5.6% of patients develop sepsis, a model that always says "no sepsis" would be 94.4% accurate — and completely useless.

Instead we use:

| Metric | What It Means | Why It Matters |
|---|---|---|
| **AUC-ROC** | How well the model ranks sick patients above healthy ones | Standard ML benchmark |
| **AUPRC** | Precision-Recall performance under class imbalance | Better than AUC when data is imbalanced |
| **Sensitivity (Recall)** | Of all patients who got sepsis, how many did we catch? | Missing a sepsis case is catastrophic |
| **Specificity** | Of all healthy patients, how many did we correctly leave alone? | Too many false alarms = alert fatigue |
| **Clinical Utility Score** | The official PhysioNet competition score that rewards early correct predictions | Lets us compare directly to published results |

---

## What Makes This Project Original

Most published studies test a new model or a new preprocessing method — but they change both at the same time and just report the final number. You cannot tell what actually drove the improvement.

**We are the first (in our course, and unusually in general) to design a controlled experiment that isolates these two factors on this dataset.** The result will tell researchers and clinicians not just *what works*, but *why it works* — which is far more useful for building real clinical tools.

---

## What We Are NOT Doing

To keep the project focused and completable, we are not:
- Building a real hospital system or clinical tool
- Testing on multiple datasets beyond PhysioNet 2019
- Performing subgroup analyses by age or demographics (acknowledged as a limitation)
- Claiming our model is ready for clinical deployment

---

## Summary in Three Sentences

We are predicting sepsis 6 hours before it happens using ICU patient data. We designed a controlled experiment that separates the effect of data preparation from the effect of model complexity — something most existing research does not do. Our goal is to produce clear, defensible evidence for which design choice matters more, so future clinical AI tools are built on solid ground.

---

## Quick Reference Card

| Question | Answer |
|---|---|
| What disease? | Sepsis |
| What task? | Predict sepsis 6 hours early |
| What data? | PhysioNet 2019 — 40,336 ICU patients |
| What experiment? | 2×2: 2 data prep strategies × 2 model types |
| What is Strategy A? | Fill missing values with the median |
| What is Strategy B? | Forward-fill + add missingness indicator columns |
| What are the simple models? | Logistic Regression and XGBoost |
| What is the advanced model? | LSTM (reads patient timeline as a sequence) |
| What is our hypothesis? | Better data prep matters more than a fancier model |
| How do we measure success? | AUC-ROC, AUPRC, Sensitivity, Specificity, Utility Score |
| What makes this original? | Controlled isolation of two design choices — rarely done |
