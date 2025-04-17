#Open-Domain-Chatbot-with-RAG

An open-domain conversational chatbot trained on the WoW (Wizard of Wikipedia) and BST (Blended Skill Talk) datasets, progressively enhanced with:
1. LSTM Seq2Seq (Baseline)
2. Attention Mechanisms
3. Transformer Architecture
4. GPT Fine-Tuning
5. Retrieval-Augmented Generation (RAG)

Features:
- Knowledge Grounding: Uses WoW for factual responses
- Emotional Intelligence: Fine-tuned on BST for natural, emotionally-aware replies
- RAG Pipeline: Dynamic knowledge retrieval from MongoDB

Repository Structure:
|--data/                 # Preprocessed datasets (WoW + BST)
|--notebook/             # EDA, prototyping (Colab/Jupyter)
|--src/
|--|--dataset.py         # Data loading and preprocessing
|--|--models/            # Model architectures (LSTM -> Transformer -> GPT)
|--|--train.py           # Training scripts
|--|--inference.py       # Chatbot GUI + RAG integration
|--configs/              # Hyperparameter configurations
|--requirements.txt      # Python dependencies
|--vocabulary.pkl        # Saved vocabulary (generated during preprocessing)
|--.gitignore            # Files to exclude from version control
|--README.md             # Project documentation

Setup:
1. Clone the repo:
   git clone https://github.com/your-username/open-domain-chatbot-with-rag.git
   cd Open-Domain-Chatbot-with-RAG

2. Install dependencies (coming soon):
   pip install -r requirements.txt



Note: This README will be updated with:
- Detailed training instructions
- Evaluation metrics (BLEU, sentiment scores)
- Demo of the chatbot in action
