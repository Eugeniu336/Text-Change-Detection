Text Change Detection
This project trains and evaluates a Support Vector Machine (SVM) model to detect changes between consecutive paragraphs in text documents. It uses TF-IDF vectorization to extract textual features and classifies whether a change has occurred between two text segments.

🚀 Project Overview
The model is trained on three difficulty levels (easy, medium, hard) using labeled datasets. Each dataset consists of pairs of consecutive paragraphs, along with ground truth labels indicating whether a change occurs.

🔹 Main Features:
 Loads and processes training and validation data from structured text files.
 Uses TF-IDF Vectorizer to extract textual features.
 Trains an SVM (Support Vector Machine) classifier for each dataset.
 Evaluates performance using classification metrics.
 Saves the evaluation results for different datasets in separate text files.

🛠 Installation & Setup
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/Text-Change-Detection.git
cd Text-Change-Detection
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the training script

bash
Copy
Edit
python train.py
📊 Dataset Structure
The dataset is organized into three categories:

bash
Copy
Edit
/easy
    /train
        text1.txt
        truth-text1.json
    /validation
        text2.txt
        truth-text2.json
/medium
/hard
Each truth-*.json file contains labels for paragraph transitions.

📈 Model Evaluation
The trained models are evaluated on the validation sets, and the classification reports are saved as:

Copy
Edit
easy_results.txt  
medium_results.txt  
hard_results.txt  
These files contain precision, recall, and F1-score for model performance analysis.

 Future Improvements
 Try other classifiers like Random Forest or Deep Learning models.
 Experiment with word embeddings instead of TF-IDF for better text representation.
 Optimize hyperparameters for better classification accuracy.

