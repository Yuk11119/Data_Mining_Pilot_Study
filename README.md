# Short Answer Grading System

An automated short answer grading system based on pre-trained BERT embeddings to support efficient assessment of students' responses.

## Repository Structure

```
.
├── Dataset/                        # Raw dataset
│   └── saf_communication_networks_english/  # SAF dataset
│       └── data/                   # Parquet files with question-answer pairs
├── Short_Answer_Grading/           # Main project code
│   ├── data/                       # Processed data directory
│   │   └── processed/              # CSV files for train/val/test splits
│   ├── results/                    # Evaluation results and visualizations  
│   ├── src/                        # Source code
│   │   ├── data_utils.py           # Data loading and processing 
│   │   ├── model.py                # BERT embedding model implementation
│   │   ├── scoring.py              # Similarity-based scoring system
│   │   ├── evaluate.py             # Evaluation metrics calculation
│   │   ├── visualization.py        # Visualization utilities
│   │   ├── error_analysis.py       # Error analysis functions
│   │   ├── model_evaluation.py     # Model evaluation scripts
│   │   └── exploration.py          # Data exploration utilities
│   ├── main.py                     # Main program entry point
│   ├── README.md                   # Project documentation
│   └── requirements.txt            # Python dependencies
└── .gitignore                      # Git ignore file
```

## Project Overview

The system uses a Sentence-BERT model (specifically `paraphrase-MiniLM-L6-v2`) to generate semantic embeddings for student and reference answers. It automatically evaluates the correctness of student answers by calculating the cosine similarity between these embeddings. The scoring system uses the following threshold-based rules:

* Similarity ≥ 0.85 → Score 2 (Completely Correct)
* 0.60 ≤ Similarity < 0.85 → Score 1 (Partially Correct)
* Similarity < 0.60 → Score 0 (Incorrect)

## Dataset

This project uses the SAF Communication Networks English dataset (Short Answer Feedback - Communication Networks - English), which contains university-level short answer questions on communication networks topics and their scores.

## Installation and Setup

1. Clone this repository
   ```
   git clone https://github.com/[your-username]/Short-Answer-Grading.git
   cd Short-Answer-Grading
   ```

2. Install required dependencies:
   ```
   pip install -r Short_Answer_Grading/requirements.txt
   ```
   
   Key dependencies include:
   ```
   sentence-transformers==2.2.2
   torch==2.0.1
   transformers==4.30.0
   scikit-learn
   matplotlib
   seaborn
   pandas
   ```

3. Download the dataset (if not already included) or use your own dataset with a similar format.

4. Navigate to the main project directory:
   ```
   cd Short_Answer_Grading
   ```

5. Run the main program:
   ```
   python main.py [options]
   ```

## Usage Examples

The system provides several command-line options for different functionalities:

1. Preprocess data:
   ```
   python main.py --preprocess
   ```

2. Evaluate model on test data:
   ```
   python main.py --evaluate
   ```

3. Analyze prediction errors:
   ```
   python main.py --analyze_errors
   ```

4. Complete workflow with visualization:
   ```
   python main.py --preprocess --evaluate --visualize
   ```

View all available options with:
```
python main.py --help
```

## Features

- **Semantic Similarity**: Utilizes pre-trained Sentence-BERT models to calculate semantic similarity between reference and student answers
- **Threshold-based Scoring**: Assigns scores based on configurable similarity thresholds
- **Detailed Error Analysis**: Identification and categorization of error patterns
- **Visualization Tools**: Various visualization methods for similarity distribution, confusion matrix, etc.

## Evaluation Metrics

The system uses the following metrics for evaluation:
- Accuracy: Proportion of correctly classified answers
- Macro-averaged F1-score: Harmonic mean of precision and recall across all classes
- Macro-averaged F2-score: Weighted average favoring recall over precision

## Detailed Documentation

For more detailed information about the implementation, features, and results, please refer to the [project documentation](Short_Answer_Grading/README.md) in the Short_Answer_Grading directory. 