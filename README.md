# Disease_Classifier
Utilising text classification techniques and machine learning algorithms to build a symptom-to-disease classifier, showcasing skills in natural language processing, feature extraction, model training, hyperparameter tuning and evaluation.

This repository contains code for building a symptom-to-disease classifier using machine learning techniques. Given a symptom description, the classifier predicts the most likely associated disease.

## Dataset

The dataset used for training and evaluation is stored in the file `Symptom2Disease.csv`. It comprises a collection of symptom descriptions and their corresponding disease labels. The dataset comprises 24 different diseases; each disease has 50 symptom descriptions, resulting in a total of 1200 datapoints

## Getting Started

1. Clone the repository:

```
git clone https://github.com/your-username/Symptom2DiseaseClassifier.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the main script to train the classifier and make predictions:

```
python main.py
```

## Project Structure

The project structure is organised as:

- `main.py`: The main script for training the classifier and making predictions.
- `exploratory_data_analysis.py`: Script for performing exploratory data analysis on the dataset.
- `utils.py`: Utility functions for data preprocessing and evaluation.
- `Symptom2Disease.csv`: Dataset file containing symptom descriptions and disease labels.

## Results

The trained classifier achieves an accuracy of 95% on the test set. The best hyperparameters were found using grid search with cross-validation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
