# Scikit-Learn

Steps to work on Data Science projects
1. Read Data to understand Data. ( plot graphs , data types etc.)
2. Create Training and Testing set for your data.
3. Feed Data to model
4. Evaluate models.

Important Notes:
- Allways try to insert data. don't try to delete colum. Just ignor it while training.

![Alt Text](https://scikit-learn.org/stable/_static/ml_map.png)


# Available Sklearn models

1. **Linear Models:**
   - Linear Regression - Mostly User
   - Ridge Regression
   - Lasso Regression
   - Logistic Regression - Mostly User to predict 0-1

2. **Support Vector Machines (SVM):**
   - SVM for classification
   - SVM for regression

3. **Tree-based Models:**
   - Decision Trees
   - Random Forest
   - Gradient Boosting (e.g., GradientBoostingClassifier, GradientBoostingRegressor)

4. **Nearest Neighbors:**
   - K-Nearest Neighbors (KNN) - Mostly User

5. **Clustering:**
   - K-Means
   - Hierarchical clustering

6. **Naive Bayes:**
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes

7. **Ensemble Methods:**
   - Voting Classifier
   - Bagging Classifier
   - AdaBoost

8. **Dimensionality Reduction:**
   - Principal Component Analysis (PCA)
   - t-Distributed Stochastic Neighbor Embedding (t-SNE)

9. **Neural Network Models:**
   - Multi-layer Perceptron (MLP)

10. **Preprocessing and Feature Selection:**
    - StandardScaler
    - MinMaxScaler
    - PolynomialFeatures
    - SelectKBest, etc.

11. **Model Selection and Evaluation:**
    - Train-Test Split
    - Cross-Validation
    - GridSearchCV

12. **Other Models:**
    - Gaussian Mixture Models (GMM)
    - Isolation Forest
    - One-Class SVM
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)


--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

1. **Linear Regression:**
   - **Use Case:** Predicting a continuous target variable based on one or more independent features.
   - **When to Choose:** When there is a linear relationship between the features and the target variable.

2. **Logistic Regression:**
   - **Use Case:** Binary classification problems where the target variable has two classes (0 or 1).
   - **When to Choose:** When you need probabilities of class membership and the decision boundary is expected to be linear.

3. **Random Forest:**
   - **Use Case:** Classification or regression tasks.
   - **When to Choose:** When you want a robust model that handles non-linearity, captures interactions between features, and is less prone to overfitting.

4. **Gradient Boosting (e.g., GradientBoostingClassifier, GradientBoostingRegressor):**
   - **Use Case:** Classification or regression tasks.
   - **When to Choose:** When you want to build a strong predictive model by combining weak learners sequentially, minimizing errors at each step.

5. **Support Vector Machines (SVM):**
   - **Use Case:** Classification or regression tasks.
   - **When to Choose:** When you need a model that finds the optimal hyperplane that maximally separates classes in feature space.

6. **K-Nearest Neighbors (KNN):**
   - **Use Case:** Classification or regression tasks.
   - **When to Choose:** When instances of the same class are close to each other in feature space, and you want predictions based on the majority class of nearby instances.

7. **K-Means:**
   - **Use Case:** Clustering data into k groups.
   - **When to Choose:** When you want to partition your data into clusters based on similarity.

8. **Decision Trees:**
   - **Use Case:** Classification or regression tasks.
   - **When to Choose:** When you want a model that recursively splits data based on the most significant feature to create a tree-like structure.

9. **Naive Bayes (e.g., Gaussian Naive Bayes, Multinomial Naive Bayes):**
   - **Use Case:** Classification tasks, especially in text classification.
   - **When to Choose:** When you want a simple probabilistic model based on the Bayes' theorem assumption.

10. **Principal Component Analysis (PCA):**
    - **Use Case:** Dimensionality reduction.
    - **When to Choose:** When you have a high-dimensional dataset and want to reduce the number of features while preserving the most important information.

The choice of model depends on factors such as the nature of your data, the size of your dataset, interpretability requirements, and the specific problem you are solving. It's often a good practice to try multiple models and evaluate their performance to determine the most suitable one for your task.