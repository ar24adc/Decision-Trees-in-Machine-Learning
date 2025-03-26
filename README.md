# Heart Disease Prediction using Decision Trees

This repository contains the implementation of a **Decision Tree** model to predict the presence of heart disease in patients based on various medical features. The dataset used in this project is the **Heart Disease dataset** from OpenML. The project demonstrates the application of Decision Trees for classification, model evaluation, and visualization.

## Project Overview

The goal of this project is to build and evaluate a machine learning model to predict whether a patient has heart disease or not. We use a **Decision Tree Classifier** from scikit-learn to train the model. The project includes data preprocessing, model training, performance evaluation, and visualization of results.

### Features of the Project:
- **Data Preprocessing**: Loading and splitting the dataset into training and testing sets.
- **Model Training**: Training a Decision Tree Classifier without pruning.
- **Model Evaluation**: Using accuracy, confusion matrix, and feature importance to assess the model’s performance.
- **Visualization**: Visualizing the trained Decision Tree and feature importance.
- **GitHub Repository**: The full code and analysis are available in this repository, allowing you to run, modify, and extend the project.

## Requirements

To run this project, you'll need the following Python libraries:

- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/heart-disease-decision-tree.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-decision-tree
   ```
3. Run the Jupyter Notebook or Python script to execute the model:
   - Open and run the notebook `heart_disease_decision_tree.ipynb` or execute the Python script `decision_tree_model.py`.

## Key Components

- **`heart_disease_decision_tree.ipynb`**: The Jupyter Notebook that contains the entire implementation, including data preprocessing, model training, and evaluation.
- **`decision_tree_model.py`**: A Python script version of the model implementation.
- **`decision_tree.png`**: Visualization of the trained Decision Tree model saved as an image.
- **`requirements.txt`**: A file listing all necessary Python libraries to run the project.

## Model Evaluation

The model was evaluated based on the following metrics:

1. **Accuracy**: The model achieved a 74% accuracy on the test dataset.
2. **Confusion Matrix**: Detailed performance breakdown with true positives, false positives, true negatives, and false negatives.
3. **Feature Importance**: Analyzed the most influential features such as the number of major vessels (`ca`), thalassemia condition (`thal`), chest pain type (`cp`), and ST depression (`oldpeak`).
4. **Decision Tree Visualization**: Visualized the structure of the trained Decision Tree, highlighting key decision points.

## Visualizations

- **Decision Tree Plot**: A plot of the decision tree structure with feature names and class labels.
- **Feature Importance Plot**: A bar chart showing the importance of each feature in the model’s decision-making process.
- **Confusion Matrix Plot**: A plot of the confusion matrix to assess model performance visually.

## Results

- The decision tree classifier provides reasonable predictions, achieving **74 percent accuracy** on the test dataset.
- Further improvements can be made through hyperparameter tuning, alternative algorithms, or additional feature engineering.
  
## Conclusion

This project demonstrates the application of **Decision Trees** in healthcare data classification. While the model performs well, further work can improve its accuracy and precision. The full code, visualizations, and results are available in this repository, allowing you to replicate or build on this work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Heart Disease dataset** from OpenML
- **scikit-learn** for machine learning implementation
- **Matplotlib** and **Seaborn** for visualizations
```

### Key Sections:
- **Project Overview**: Briefly explains what the project is about.
- **Requirements**: Lists the libraries needed to run the project.
- **Usage**: Provides step-by-step instructions for cloning and running the project.
- **Key Components**: Explains the contents of the repository.
- **Model Evaluation**: Describes the evaluation metrics and how the model performed.
- **Visualizations**: Mentions the visualizations used to evaluate and interpret the model.
- **Conclusion**: Summarizes the findings and suggests areas for improvement.

Make sure to adjust any file names or URLs specific to your project. This README will guide users through understanding the project and how to use the repository.
