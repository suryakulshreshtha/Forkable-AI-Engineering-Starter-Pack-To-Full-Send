# Full Send

The capstone: PocketRAG, a question-answering tool for this very repository. You load its Markdown files, split them into sections and chunks, rank chunks with the BM25 you already built, compose an answer only from sentences you can cite, refuse when the docs do not cover the question, and grade the whole system against a golden question set. Then `python levels/L9-full-send/app.py "your question"` answers questions about the repo, using your code.
