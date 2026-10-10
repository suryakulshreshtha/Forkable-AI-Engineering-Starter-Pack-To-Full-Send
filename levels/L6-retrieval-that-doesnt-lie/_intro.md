# Retrieval That Doesn't Lie

A model that answers from memory will confidently invent things. A model that answers from retrieved text can at least be checked. This level builds the honest half of retrieval-augmented generation: cutting documents into chunks, ranking them with BM25, fusing several rankings, and measuring whether the right evidence actually came back.
