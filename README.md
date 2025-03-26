# Emotion Detection Project

## Overview
This project aims to detect and classify emotions in text data using various machine learning and deep learning approaches. The dataset consists of English Twitter messages meticulously annotated with six fundamental emotions: anger, fear, joy, love, sadness, and surprise.

## Dataset
The "Emotions" dataset contains Twitter messages labeled with the following emotions:
- Sadness (0)
- Joy (1)
- Love (2)
- Anger (3)
- Fear (4)
- Surprise (5)

The dataset is relatively large, containing 416,809 entries, but exhibits class imbalance, with "surprise" being the least represented emotion.

## Features
- **Text preprocessing pipeline** including tokenization, stop word removal, and stemming
- **Exploratory data analysis** with visualizations of class distributions
- **Data balancing techniques** to address class imbalance issues
- **Feature extraction** using various methods:
  - Bag of Words
  - TF-IDF
  - Word embeddings
- **Multiple machine learning models**:
  - RandomForest
  - SVC (Support Vector Classifier)
  - Linear SVC
  - KNeighbors
  - AdaBoost
  - XGBoost
- **Deep learning approaches** using Keras/TensorFlow:
  - CNN (Convolutional Neural Network)
  - Various neural network architectures
- **Hyperparameter tuning** using Optuna

## Requirements
The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- nltk
- gensim
- xgboost
- wordcloud
- textblob
- optuna

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras nltk gensim xgboost wordcloud textblob optuna
```

## Usage
1. Clone the repository
```bash
git clone https://github.com/AviRahimov/Emotion_detection-ML-project.git
cd Emotion_detection-ML-project
```

2. Open and run the Jupyter notebook
```bash
jupyter notebook Emotions_Detection_Project.ipynb
```

Alternatively, you can open the notebook in Google Colab.

## Project Structure
- `Emotions_Detection_Project.ipynb`: Main notebook containing all the code, analysis, and models
- The project follows these steps:
  1. Data loading and exploration
  2. Text preprocessing
  3. Feature extraction
  4. Model training and evaluation
  5. Hyperparameter tuning
  6. Results analysis

## Results
The project compares the performance of various machine learning and deep learning models on the emotion classification task, evaluating them using metrics such as accuracy, precision, recall, and F1-score.

## Future Work
- Implement more advanced NLP techniques like BERT or transformer-based models
- Explore multi-modal emotion detection by incorporating other data sources
- Develop a web application for real-time emotion detection from text input

## Author
- [Avi Rahimov](https://github.com/AviRahimov)

## License
This project is open source and available under the [MIT License](LICENSE).
