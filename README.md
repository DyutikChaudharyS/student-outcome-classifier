# Student Outcome Classification

Machine Learning Assignment 2 | M.Tech (Data Science and Engineering)

## a. Problem statement

This assignment compares five supervised learning algorithms for predicting student outcomes: Dropout, Enrolled and Graduate. The objective is to evaluate how well the models distinguish these classes using demographic, financial and academic information.

Class imbalance is an important part of this problem. Accuracy alone can hide poor performance on the smaller classes, so the comparison also includes macro precision, recall, F1, ROC AUC and Matthews correlation coefficient (MCC).

## b. Dataset description

The project uses the *Predict Students' Dropout and Academic Success* dataset from the UCI Machine Learning Repository. The local dataset contains 4,424 records, 36 input features and a target column named `Target`.

| Target class | Number of records | Percentage |
|---|---:|---:|
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |
| Graduate | 2,209 | 49.9% |

The features include age at enrolment, application details, parents' qualifications and occupations, tuition fee status, scholarship status, first- and second-semester academic results, and economic indicators. The local dataset has no missing values.

Although the attributes are stored numerically, nine are treated as coded categories rather than continuous measurements. These are marital status, application mode, course, previous qualification, nationality, and the qualifications and occupations of both parents.

### Data preparation

The data is divided into training and test sets using an 80:20 stratified split with `random_state=42`. This produces 3,539 training records and 885 test records. The test partition is saved as `test_data.csv`.

The nine categorical columns are one-hot encoded, with categories occurring fewer than ten times grouped together. The remaining 27 columns are standardised. Preprocessing is included in each model's pipeline so that it is fitted on the training portion of each grid-search fold.

## c. Repository and project structure

Repository: https://github.com/DyutikChaudharyS/student-outcome-classifier

```text
student-outcome-classifier/
|-- app.py                       Streamlit application
|-- README.md                    Assignment report and usage instructions
|-- requirements.txt            Python dependencies
|-- test_data.csv                Test partition
|-- data/
|   |-- students_dropout_academic_success.csv
|-- model/
|   |-- common.py                Shared constants and evaluation functions
|   |-- train_models.py          Model training and parameter search
|   |-- *.joblib                 Five fitted model pipelines
|   |-- test_metrics.csv         Saved test results
|   |-- model_metadata.json      Parameters, CV scores and library version
|-- .streamlit/
    |-- config.toml              Application configuration
```

## d. Methodology and results

The five classifiers are Logistic Regression, Decision Tree, k-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine (SVM). Each model uses the same training and test partition.

Hyperparameters are selected using five-fold stratified cross-validation on the training set, with macro F1 as the scoring criterion. The selected pipeline for each classifier is then evaluated on the held-out test set. The SVM uses an RBF kernel and sigmoid calibration to obtain class probabilities.

### Selected hyperparameters

| Model | Parameters | CV macro F1 |
|---|---|---:|
| Logistic Regression | C = 0.1, balanced class weights | 0.7159 |
| Decision Tree | Gini criterion, maximum depth = 8, minimum samples per leaf = 25 | 0.6732 |
| kNN | 11 neighbours, uniform weights | 0.6170 |
| Gaussian Naive Bayes | var_smoothing = 0.1 | 0.6252 |
| SVM | RBF kernel, C = 10, gamma = 0.005, balanced class weights | 0.7274 |

### Evaluation metrics

Precision, recall and F1 are macro-averaged, giving equal weight to each target class. AUC is calculated separately for each class against the remaining classes and then averaged. MCC measures agreement between predicted and actual labels using the full confusion matrix.

For uploaded data that does not contain all three classes, the app averages AUC over the classes for which both positive and negative examples exist. AUC is unavailable if no class meets this condition. Precision, recall and F1 still include all three labels, with undefined values set to zero.

### Test results

The following values are taken from `model/test_metrics.csv` and refer to the 885-record test partition. The highest value for each metric is shown in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7345 | **0.8843** | 0.7071 | 0.7125 | 0.6986 | 0.5900 |
| Decision Tree | 0.7390 | 0.8488 | 0.6775 | 0.6629 | 0.6683 | 0.5687 |
| kNN | 0.7164 | 0.8238 | 0.6575 | 0.6113 | 0.6201 | 0.5261 |
| Naive Bayes | 0.6893 | 0.8105 | 0.6297 | 0.6116 | 0.6176 | 0.4879 |
| Support Vector Machine | **0.7627** | 0.8635 | **0.7166** | **0.7136** | **0.7134** | **0.6158** |

### Discussion

| Model | Interpretation of the results |
|---|---|
| Logistic Regression | Achieved the highest AUC (0.8843) and a macro F1 of 0.6986. Its stronger AUC does not translate into the highest classification accuracy, since AUC evaluates probability ranking rather than the final predicted label. |
| Decision Tree | Achieved 0.7390 accuracy, slightly above Logistic Regression, but a lower macro F1 of 0.6683. The difference shows why accuracy alone is insufficient for this dataset. |
| kNN | Achieved 0.7164 accuracy and 0.6201 macro F1. Its lower macro recall indicates weaker average recall across the three classes. The effect of feature encoding on neighbour distances would require further investigation. |
| Gaussian Naive Bayes | Had the lowest accuracy, AUC and MCC in this comparison. Its distributional and conditional-independence assumptions may be restrictive for these features, but the scores alone do not establish the cause of the lower performance. |
| SVM | Achieved the highest accuracy (0.7627), macro F1 (0.7134) and MCC (0.6158), and led on five of the six test metrics. It also had the highest cross-validation macro F1. |

SVM gives the strongest overall result in this experiment. Logistic Regression remains a useful comparison because its macro F1 is only 0.0148 lower and its AUC is higher. These findings apply to the current split and parameter grids; they do not establish that one algorithm will perform best on every student dataset.

### Limitations

The inputs include results from both the first and second semesters. The current model therefore cannot be described as a prediction system using only information available at admission. An admission-time study would need a separate experiment that excludes later academic results.

The reported scores come from one held-out partition, without confidence intervals or repeated-split evaluation. Evaluation on data from another institution would be needed to assess how well the models generalise. The model outputs describe predicted outcomes and do not identify the causes of dropout.

## Running the application

Install the dependencies and start Streamlit from the project directory:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved pipelines are included, so retraining is optional. To repeat the training process:

```bash
python model/train_models.py
```

This command overwrites the saved pipelines, test partition, metrics and metadata. The existing models were saved using scikit-learn 1.9.1, which is pinned in `requirements.txt`. If models are retrained with another version, update the dependency pin to match.

## Using the application

The app initially evaluates the bundled test partition. A different CSV can be uploaded through the sidebar, and the classifier can be selected from the dropdown.

- Include all 36 feature columns. The sample CSV provides the expected format.
- Include `Target` to display evaluation metrics, a confusion matrix and a classification report. Without it, the app provides predictions only.
- Use the comparison tab to view the five models on the same uploaded data.
- Download predictions and class probabilities as a CSV.

Both comma- and semicolon-separated files are accepted. Headers using spaces and apostrophes are matched to the underscore-based names used during training. Uploaded training records should not be used to report independent test performance.

## Reference

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
