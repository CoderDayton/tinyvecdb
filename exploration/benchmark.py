# benchmark.py
import sqlite3
import sqlite_vec
import numpy as np
import time
import struct
import os

#
N = 10_000
DIM = 384  # common for all-MiniLM-L6-v2, BGE-small, etc.


def serialize_f32(v):
    return struct.pack("<%sf" % len(v), *v)


# start from a fresh database every run
if os.path.exists("benchmark.db"):
    os.remove("benchmark.db")

db = sqlite3.connect("benchmark.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
db.execute(
    f"CREATE VIRTUAL TABLE IF NOT EXISTS bench USING vec0(embedding float[{DIM}])"
)

vectors = np.random.randn(N, DIM).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize for cosine

t0 = time.time()
with db:
    params = [(serialize_f32(vectors[i]),) for i in range(N)]
    db.executemany("INSERT INTO bench(embedding) VALUES (?)", params)
insert_time = time.time() - t0
print(
    f"Inserted {N:,} × {DIM}-dim vectors in {insert_time:.2f}s ({N / insert_time:,.0f} vec/s)"
)

file_size = os.path.getsize("benchmark.db") / 1024 / 1024
print(f"DB size: {file_size:.1f} MB")

query = np.random.randn(DIM).astype(np.float32)
query /= np.linalg.norm(query)

t0 = time.time()
for _ in range(100):
    db.execute(
        """
        SELECT rowid, distance FROM bench
        WHERE embedding MATCH ?
        ORDER BY distance LIMIT 10
    """,
        [serialize_f32(query)],
    )
query_time = (time.time() - t0) / 100
print(f"Average query time (k=10): {query_time * 1000:.2f} ms")
