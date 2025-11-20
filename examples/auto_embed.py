from tinyvecdb import VectorDB

db = VectorDB(":memory:")
db.add_texts(["Paris is beautiful", "Berlin has great beer"])  # auto-embeds!

results = db.similarity_search("Where should I drink beer?", k=1)
print(results[0][0].page_content)  # → Berlin has great beer
