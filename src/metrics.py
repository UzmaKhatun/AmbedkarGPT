import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

# Download tokenizer for BLEU
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class RAGMetrics:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_retrieval_metrics(self, retrieved_docs, expected_sources, k=3):
        """
        Calculates Hit Rate, MRR, and Precision@K
        """
        if not expected_sources: # For unanswerable questions
            return {"hit_rate": 1.0 if not retrieved_docs else 0.0, "mrr": 1.0, "precision": 1.0}

        hits = 0
        reciprocal_rank = 0
        
        # Clean expected sources to match filename only
        expected_files = [os.path.basename(s) for s in expected_sources]
        
        # Check retrieved docs
        retrieved_files = []
        for doc in retrieved_docs:
            # Handle the Source tag we added in ingestion
            # Metadata source usually is full path, get basename
            meta_source = doc.metadata.get('source', '')
            retrieved_files.append(os.path.basename(meta_source))

        # Hit Rate (Is any expected doc in retrieved?)
        is_hit = any(f in expected_files for f in retrieved_files)
        
        # MRR (1 / Rank of first correct doc)
        for i, file in enumerate(retrieved_files):
            if file in expected_files:
                reciprocal_rank = 1 / (i + 1)
                break
        
        # Precision (How many retrieved are correct / Total retrieved)
        correct_retrieved = sum(1 for f in retrieved_files if f in expected_files)
        precision = correct_retrieved / len(retrieved_files) if retrieved_files else 0

        return {
            "hit_rate": 1.0 if is_hit else 0.0,
            "mrr": reciprocal_rank,
            "precision": precision
        }

    def calculate_semantic_scores(self, generated_answer, ground_truth):
        """
        Calculates Cosine Similarity, BLEU, and ROUGE-L
        """
        # 1. Cosine Similarity
        vec_gen = self.embedding_model.embed_query(generated_answer)
        vec_truth = self.embedding_model.embed_query(ground_truth)
        cosine_sim = cosine_similarity([vec_gen], [vec_truth])[0][0]

        # 2. ROUGE-L
        rouge_scores = self.rouge.score(ground_truth, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure

        # 3. BLEU
        ref_tokens = word_tokenize(ground_truth.lower())
        cand_tokens = word_tokenize(generated_answer.lower())
        # Use weights for 1-gram (simple overlap) to avoid 0 score for short sentences
        bleu = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))

        return {
            "cosine_similarity": float(cosine_sim),
            "rouge_l": float(rouge_l),
            "bleu_score": float(bleu)
        }