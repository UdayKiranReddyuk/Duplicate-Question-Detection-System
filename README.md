# Automatic Duplicate Question Detection (TF‑IDF + LDA)

This project detects duplicate or semantically similar questions using a combination of TF‑IDF cosine similarity and LDA topic modelling on historical Q&A data. 
It is designed to be lightweight and runnable on a typical laptop while still demonstrating core NLP concepts.

---

## Project Overview

*Goal*: Automatically identify past questions that are duplicates or near-duplicates of a new question.
*Techniques*:
  - Text cleaning with NLTK (tokenisation, stopword removal, stemming).
  - TF‑IDF vectorisation with cosine similarity for lexical similarity.
  - Gensim LDA topic modelling for semantic similarity.
  - Weighted fusion of TF‑IDF and LDA similarities.
*Evaluation*: Recall@K (K = 1, 5, 10) to measure how often the true duplicate appears in the top‑K retrieved past questions.

---

## Dataset

- Input file: "filename.csv"
- Expected columns:
  - 'PastQuesTitle', 'PastQuesBody', 'PastQuesTags'
  - 'DuplicateQuesTitle', 'DuplicateQuesBody', 'DuplicateQuesTags'
- The script uses:
  - 'N1' past questions (rows from the *Past* columns).
  - 'N2' duplicate questions (rows from the *Duplicate* columns).
- You can adjust 'N1', 'N2', and 'NUM_TOPICS' at the top of the script to balance speed vs. accuracy for your machine.

> Note: Ensure the dataset is placed under 'data/filename.csv' or update the path in the script accordingly.

---

## Example Output

✔ Preprocessing completed
✔ TF-IDF similarity computed  
✔ LDA similarity computed
Recall@1: 0.45
Recall@5: 0.78
Recall@10: 0.92
✔ Execution completed successfully

---

## References

  - Latent Dirichlet Allocation - Journal of Machine Learning Research[https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf]

  - Cosine Similarity - Scikit-learn Documentation[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html]
