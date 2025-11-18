# basic_demo.py
import sqlite3
import sqlite_vec
from typing import List
import struct
import time


def serialize_f32(vector: List[float]) -> bytes:
    """sqlite-vec expects raw little-endian float32 blobs"""
    return struct.pack("<%sf" % len(vector), *vector)


def serialize_i8(vector: List[int]) -> bytes:
    """Convert Python ints (0-255 or -128..127) into signed int8 bytes.

    Note: for `int8[]` columns and MATCH queries, sqlite-vec expects
    its own int8 vector type, so we will wrap this with `vec_int8(?)`
    in SQL, not pass the raw blob directly.
    """
    signed = [(int(v) + 128) % 256 - 128 for v in vector]
    return struct.pack("<%sb" % len(signed), *signed)


def serialize_bits(vector: List[int]) -> bytes:
    """Pack 0/1 integers into little-endian bitfield bytes."""
    byte_count = (len(vector) + 7) // 8
    buf = bytearray(byte_count)
    for idx, bit in enumerate(vector):
        if bit not in (0, 1):
            raise ValueError("Bit vectors must only contain 0 or 1 values")
        if bit:
            buf[idx // 8] |= 1 << (idx % 8)
    return bytes(buf)


# 1. Connect + load extension
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

print("SQLite version:", db.execute("select sqlite_version()").fetchone()[0])
print("sqlite-vec version:", db.execute("select vec_version()").fetchone()[0])

# 2. Create table with different column types
db.execute("""
    CREATE VIRTUAL TABLE vec_example USING vec0(
        rowid INTEGER PRIMARY KEY,
        embedding float[4],          -- classic float32 vectors
        embedding_int8 int8[4],      -- quantized int8 (4× smaller)
        embedding_bit bit[8],        -- binary quantized (32× smaller!)
        category TEXT,               -- regular metadata column
        tags TEXT                    -- another metadata column
    )
""")

# 3. Sample data
items = [
    (1, [0.1, 0.1, 0.1, 0.1], [25, 25, 25, 25], [0, 0, 0, 0], "fruit", "sweet"),
    (2, [0.2, 0.2, 0.2, 0.2], [51, 51, 51, 51], [0, 0, 0, 1], "vegetable", "crunchy"),
    (3, [0.9, 0.9, 0.9, 0.9], [230, 230, 230, 230], [1, 1, 1, 1], "fruit", "sweet"),
    (4, [0.8, 0.8, 0.8, 0.8], [204, 204, 204, 204], [1, 1, 1, 0], "fruit", "tart"),
]

query_vec = [0.85, 0.85, 0.85, 0.85]

# 4. Insert everything
start = time.time()
with db:
    for row in items:
        db.execute(
            """
            INSERT INTO vec_example(rowid, embedding, embedding_int8, embedding_bit, category, tags)
            VALUES (?, ?, vec_int8(?), vec_bit(?), ?, ?)
        """,
            [
                row[0],
                serialize_f32(row[1]),
                serialize_i8(row[2]),
                serialize_bits(row[3]),
                row[4],
                row[5],
            ],
        )
print(f"Insert time: {time.time() - start:.3f}s")

# 5. Basic cosine search
rows = db.execute(
    """
    SELECT
        rowid,
        category,
        distance
    FROM vec_example
    WHERE embedding MATCH ?
    AND k = 3
    ORDER BY distance
""",
    [serialize_f32(query_vec)],
).fetchall()

print("\nTop-3 cosine results:")
for r in rows:
    print(r)

# 6. Try int8 quantized search (huge storage win, tiny accuracy loss)
rows_int8 = db.execute(
    """
    SELECT
        rowid,
        category,
        distance
    FROM vec_example
    WHERE embedding_int8 MATCH vec_int8(?)
    AND k = 3
    ORDER BY distance
""",
    [serialize_i8(query_vec)],
).fetchall()
print("\nTop-3 int8 quantized results:")
for r in rows_int8:
    print(r)

# 7. Try bit quantized search (massive storage win, bigger accuracy loss)
rows_bit = db.execute(
    """
    SELECT
        rowid,
        category,
        distance
    FROM vec_example
    WHERE embedding_bit MATCH vec_bit(?)
    AND k = 3
    ORDER BY distance
""",
    [serialize_bits([1 if v >= 0.5 else 0 for v in query_vec])],
).fetchall()
print("\nTop-3 bit quantized results:")
for r in rows_bit:
    print(r)

# 8. Metadata usage (no filtering yet, but storage works)
rows_meta = db.execute(
    "SELECT rowid, category, tags FROM vec_example WHERE category = 'fruit'"
).fetchall()
print("\nMetadata filtering (plain SQL):", rows_meta)
