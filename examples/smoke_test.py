# examples/smoke_test.py
from tinyvecdb import VectorDB

db = VectorDB(":memory:")

embeddings = [
    [0.1, 0.2, 0.9],  # apple
    [0.8, 0.1, 0.3],  # banana
    [0.9, 0.8, 0.1],  # orange
]

db.add_texts(
    texts=["apple", "banana", "orange"],
    embeddings=embeddings,
    metadatas=[{"type": "fruit"}, {"type": "fruit"}, {"type": "fruit"}],
)

results = db.similarity_search([0.95, 0.95, 0.95], k=2)

for doc, score in results:
    print(f"{score:.4f} → {doc.page_content} {doc.metadata}")

# With filter
results_filtered = db.similarity_search(
    [0.95, 0.95, 0.95], k=2, filter={"type": "fruit"}
)
print("\nFiltered:")
for doc, score in results_filtered:
    print(f"{score:.4f} → {doc.page_content} {doc.metadata}")
