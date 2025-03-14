import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Definim o functie pentru incarcarea datelor
def load_data(path, is_train=True):
    data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r', newline="", encoding='utf-8') as file:
                text_data = file.read()
            if is_train:
                with open(os.path.join(path, 'truth-' + filename.replace('.txt', '.json')), 'r', encoding='utf-8') as file:
                    ground_truth = json.load(file)
            else:
                with open(os.path.join(path.replace('train', 'validation'), 'truth-' + filename.replace('.txt', '.json')), 'r', encoding='utf-8') as file:
                    ground_truth = json.load(file)
            paragraphs = text_data.split('\n')
            for i in range(len(paragraphs) - 1):
                if i < len(ground_truth['changes']):
                    data.append({
                        'text1': paragraphs[i],
                        'text2': paragraphs[i + 1],
                        'change': ground_truth['changes'][i]
                    })
    return data

# Load training data
train_data_easy = load_data('easy/train')
train_data_medium = load_data('medium/train')
train_data_hard = load_data('hard/train')

# Load validation data
validation_data_easy = load_data('easy/validation', is_train=False)
validation_data_medium = load_data('medium/validation', is_train=False)
validation_data_hard = load_data('hard/validation', is_train=False)

# Extragem caracteristicile folosind TfidfVectorizer
vectorizer = TfidfVectorizer()

# Antrenam si evaluam un model pentru fiecare set de date
for train_data, validation_data in [(train_data_easy, validation_data_easy), (train_data_medium, validation_data_medium), (train_data_hard, validation_data_hard)]:
    X_train = vectorizer.fit_transform([d['text1'] + ' ' + d['text2'] for d in train_data])
    y_train = [d['change'] for d in train_data]
    X_validation = vectorizer.transform([d['text1'] + ' ' + d['text2'] for d in validation_data])
    y_validation = [d['change'] for d in validation_data]

    # Antrenam un model SVC
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluam modelul pe setul de validare
    y_pred = model.predict(X_validation)
    print(classification_report(y_validation, y_pred, zero_division=0))
    # Antrenam si evaluam un model pentru fiecare set de date
    for dataset_name, train_data, validation_data in [('easy', train_data_easy, validation_data_easy),
                                                      ('medium', train_data_medium, validation_data_medium),
                                                      ('hard', train_data_hard, validation_data_hard)]:
        X_train = vectorizer.fit_transform([d['text1'] + ' ' + d['text2'] for d in train_data])
        y_train = [d['change'] for d in train_data]
        X_validation = vectorizer.transform([d['text1'] + ' ' + d['text2'] for d in validation_data])
        y_validation = [d['change'] for d in validation_data]

        # Antrenam un model SVC
        model = SVC()
        model.fit(X_train, y_train)

        # Evaluam modelul pe setul de validare
        y_pred = model.predict(X_validation)
        report = classification_report(y_validation, y_pred, zero_division=0)

        # Scriem rezultatele in fisier
        with open(f'{dataset_name}_results.txt', 'w') as f:
            f.write(report)







