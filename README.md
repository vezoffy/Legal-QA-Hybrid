# Legal Question Answering System

This project implements a hybrid **Legal Question Answering System** that combines extractive and generative approaches to answer legal queries effectively. It leverages custom BiLSTM models for extraction and fine-tuned T5 models for text generation.

## Features

- **Hybrid Architecture**: Uses Extractive (BiLSTM + Attention) and Generative (T5) models.
- **Intelligent Fallback**: Attempts to extract precise answers first; falls back to generating answers if confidence is low.
- **Legal-Specific Processing**: Includes specialized question classification (Definition, Factual, Procedure, etc.) and preprocessing.
- **Advanced Retrieval**: Utilizes Sentence Transformers and BM25 for context retrieval.

## Project Structure

- `comp5.py`: Core application logic containing model definitions (`BiLSTMAttention`), training loops, and the `AnswerExtractorGenerator` class.
- `main.ipynb`: Data experimentation notebook for preprocessing, indexing, and initial analysis.
- `generative_model/`: Directory for fine-tuned T5 model artifacts.
- `legal_qa_processed.json`: Preprocessed training dataset (Not included in repo due to size).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download necessary NLTK data (automatically handled in scripts, but can be run manually):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

### Training Models
To train the models, you can run the main script (ensure you have the dataset available):

```bash
python comp5.py
```
*Note: check the `Example usage` block at the bottom of `comp5.py` to adjust training parameters.*

### Running the Notebook
For data analysis and preprocessing steps:

```bash
jupyter notebook main.ipynb
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
- Scikit-learn
- Pandas, NumPy, NLTK

## License
[MIT](https://choosealicense.com/licenses/mit/)
