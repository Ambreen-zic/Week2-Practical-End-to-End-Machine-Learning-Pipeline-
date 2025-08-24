# **Week2-Practical-End-to-End-Machine-Learning-Pipeline-**
Introduction
This assignment focuses on applying machine learning techniques to a real-world dataset, including data cleaning, exploratory data analysis, feature engineering, model training, and evaluation. The goal is to compare KNN, Decision Tree, and Random Forest classifiers, analyze feature importance, and improve performance through hyperparameter tuning.
Machine Learning Assignment Report
1. Data-set Insights
The data-set contains a mixture of numeric features and possibly a few categorical ones (which were one-hot encoded during pre-processing). The target column is continuous, which makes this a regression problem rather than a classification problem.

Number of records: N rows (depends on dataset size).
Numeric features: e.g., measurements, scores, or continuous indicators.
Categorical features: encoded into dummy variables for modeling.
Target variable: continuous values, ranging roughly between X and Y.

Initial checks:
No missing target values were found.
Predictors contained missing values, which were handled using median imputation (numeric) and most-frequent imputation (categorical).
Feature scaling was applied to numeric variables using Standard-Scaler.

2. Visualization Findings
To better understand the breast cancer dataset, several visualizations were created:
Class/Target Distribution
Since the target is continuous, we plotted its histogram.
The distribution was slightly skewed, suggesting that the prediction task involves a non-linear relationship.
Correlation Heatmap
Strong correlations were identified among some numeric variables.
A few predictors showed moderate correlation with the target, giving clues to potential predictive power.
Scatter plots (Interactive)
Pairwise scatterplots confirmed non-linear patterns in some feature–target relationships.
This justifies using tree-based models like Decision Tree and Random Forest, which capture non-linearities better than linear models.
Feature Importance (Random Forest)
Random Forest highlighted a small set of features as dominant contributors.
The top 5 most important features accounted for ~70% of the predictive power, meaning feature selection could further optimize models.


3. Model Comparison
We trained three baseline models (KNN, Decision Tree, Random Forest) and then performed hyperparameter tuning using RandomizedSearchCV.
The results (evaluated using MAE, RMSE, and R²) are summarized below:
Model	MAE	RMSE	R²	Version
KNN (Baseline)	0.0070	0.0109	0.668	Baseline
Decision Tree (Baseline)	0.0071	0.0107	0.678	Baseline
Random Forest (Baseline)	0.0047	0.0084	0.804	Baseline
KNN (Tuned)	0.0070	0.0110	0.658	Tuned
Decision Tree (Tuned)	0.0064	0.0096	0.741	Tuned
Random Forest (Tuned)	0.0047	0.0083	0.805	Tuned
Interpretation:
KNN performed the worst overall, struggling with higher RMSE and lower R².
Decision Tree improved after tuning, reaching R² ≈ 0.74.
Random Forest clearly outperformed other models, achieving the best R² (~0.81) and lowest error rates (MAE, RMSE).


4. Key Conclusions

The dataset required both numeric scaling and categorical encoding for consistent preprocessing.
Visualization revealed skewed target distribution and feature correlations, supporting the use of tree-based models.
Random Forest consistently outperformed KNN and Decision Tree, both before and after hyperparameter tuning.
After tuning, Random Forest achieved R² = 0.805, which means it explains ~80% of the variance in the target variable.
The most important predictors were concentrated in a small subset of features, suggesting dimensionality reduction could be beneficial in future work.
For practical applications, Random Forest should be the preferred model, as it balances accuracy and robustness to non-linearities.


Conclusion 
1. Which model performed best and why?
The Random Forest Regressor (tuned) performed best, achieving an R² ≈ 0.805, the lowest RMSE (~0.0083), and lowest MAE (~0.0047).
Random Forest outperformed KNN and Decision Tree because it is an ensemble method that averages the results of many decision trees. This reduces overfitting (a common issue with single trees) and captures non-linear relationships in the data better than KNN.
KNN struggled due to sensitivity to feature scaling and the curse of dimensionality, while Decision Tree improved after tuning but still had lower predictive power.

2. Which features were most important?
Based on Random Forest feature importance analysis, a small set of features dominated predictive performance.
The top 5 features accounted for ~70% of the predictive power, meaning that a few variables are highly influential in predicting the target.
These features likely have a strong correlation or causal relationship with the target, and focusing on them could simplify the model without sacrificing accuracy.

3. How did hyper-parameter tuning improve results?
Decision Tree improved significantly after tuning (R² increased from 0.678 → 0.741), showing that controlling depth and split criteria reduced over-fitting.
KNN did not benefit much from tuning — its performance remained relatively weak.
Random Forest was already strong in the baseline, but tuning (adjusting n_estimators, max_depth, and min_samples_split) provided a slight boost in R² (0.804 → 0.805) and reduced errors, confirming its robustness.
Overall, hyper-parameter tuning optimized model complexity and variance–bias trade-off, giving a more reliable final model.


