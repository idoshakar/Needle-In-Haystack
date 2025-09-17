"""
hdbscan_duckdb_food_normalize.py

Pipeline to cluster FoodData Central entries using HDBSCAN while minimizing RAM usage by
leveraging DuckDB and disk-backed numpy memmaps.

Assumptions:
- You have a DuckDB database file with a table named `foods` (configurable) that contains at
  least these columns: `fdc_id` (unique id), `description` (text), and nutrient columns
  (e.g. `protein_g`, `fat_g`, `carbohydrate_g`, `calories_kcal`, ...). You can adapt names
  as needed.

High-level steps:
1. Count rows and prepare disk-backed memmaps for embeddings and nutrient arrays.
2. Stream rows from DuckDB in batches; compute text embeddings (SentenceTransformer) and
   write them + nutrients into the memmaps.
3. Run IncrementalPCA to reduce embedding dimensionality (disk-backed streaming) to keep RAM low.
4. Scale nutrient vectors using StandardScaler (partial_fit) and transform on-disk.
5. Concatenate reduced embeddings + scaled nutrients into a feature memmap.
6. Load the final feature matrix (much smaller) to memory and run HDBSCAN.
7. Write cluster labels back into a new DuckDB database and compute aggregated nutrient
   profiles per cluster.

Notes:
- The heaviest parts (raw embeddings) are stored on disk to avoid holding the whole matrix in RAM.
- You still need enough RAM to hold the reduced feature matrix (n_samples x (reduced_dim + n_nutrients)).
- Tune `emb_model_name`, `embed_batch_size`, `pca_components`, and `hdbscan` params for your dataset and RAM.

Dependencies (install if needed):
    pip install duckdb pandas numpy sentence-transformers scikit-learn hdbscan

Run example:
    python hdbscan_duckdb_food_normalize.py \
      --input-db foods.duckdb --input-table foods \
      --output-db foods_clustered.duckdb

"""

import argparse
import os
from tqdm import tqdm
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-db", default="food_central.db", help="Input DuckDB file path")
    p.add_argument("--input-table", default="pivoted_full_join", help="Table inside input DB containing food rows")
    p.add_argument("--output-db", default="foods.db", help="Output DuckDB file path to write clusters & summaries")
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=EMBED_SIZE, help="Expected embedding dimension of model")
    p.add_argument("--batch-size", type=int, default=128, help="DB fetch batch size")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=None)
    return p.parse_args()


def prepare_memmaps(n_rows, embed_dim, n_nutrients, work_dir):
    """Create memmap files for embeddings, nutrients, reduced embeddings, scaled nutrients, and labels."""
    emb_path = os.path.join(work_dir, "embeddings.npy")
    nut_path = os.path.join(work_dir, "nutrients.npy")
    reduced_path = os.path.join(work_dir, "emb_reduced.npy")
    scaled_nut_path = os.path.join(work_dir, "nutrients_scaled.npy")
    labels_path = os.path.join(work_dir, "labels.npy")

    embeddings = np.lib.format.open_memmap(emb_path, mode="w+", dtype="float32", shape=(n_rows, embed_dim))
    nutrients = np.lib.format.open_memmap(nut_path, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))
    emb_reduced = np.lib.format.open_memmap(reduced_path, mode="w+", dtype="float32", shape=(n_rows, 0 if n_rows==0 else 0))
    scaled_nuts = np.lib.format.open_memmap(scaled_nut_path, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype="int32", shape=(n_rows,))

    # Note: emb_reduced will be recreated later after PCA fit (we need components to set shape)
    return {
        "embeddings": embeddings,
        "nutrients": nutrients,
        "emb_reduced_path": reduced_path,
        "scaled_nuts": scaled_nuts,
        "labels": labels,
        "emb_path": emb_path,
        "nut_path": nut_path,
    }

import torch
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import re

MAX_LENGTH = 20
EMBED_SIZE = 100

def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove single char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]

def encode_glove(s):
    cleaned = tokenize(s)
    if len(cleaned) == 0:
        embedded = torch.zeros(MAX_LENGTH, EMBED_SIZE)
    else:
        embedded = embedding.get_vecs_by_tokens(cleaned, lower_case_backup=True)
    if embedded.shape[0] != MAX_LENGTH or embedded.shape[1] != EMBED_SIZE:
        embedded = torch.nn.functional.pad(embedded, (0, 0, 0, MAX_LENGTH - embedded.shape[0]))
    ret = pd.Series(embedded.mean(dim=0).numpy())
    return ret

embedding = GloVe(name='6B', dim=EMBED_SIZE)
tokenizer = get_tokenizer(tokenizer=tokenize)

def write_rows_to_memmaps(con, table, id_col, text_col, nutrient_cols, embeddings_memmap, nutrients_memmap,
                          fetch_batch):
    cursor = con.cursor()
    sql = f"SELECT {id_col}, {text_col}, \"{'", "' .join(nutrient_cols)}\" FROM {table}"
    cursor.execute(sql)

    idx = 0
    while True:
        rows = cursor.fetchmany(fetch_batch)
        if not rows:
            break
        # rows is a list of tuples: (id, description, nut1, nut2, ...)
        #ids = [r[0] for r in rows]
        texts = [r[1] if r[1] is not None else "" for r in rows]
        nuts = np.array([[None if v is None else float(v) for v in r[2:]] for r in rows], dtype="float32")

        # compute embeddings in smaller batches to avoid spikes
        emb = []
        for text in texts:
            emb.append(encode_glove(text))

        # write into memmaps
        batch_n = len(emb)
        embeddings_memmap[idx: idx + batch_n, :] = emb
        nutrients_memmap[idx: idx + batch_n, :] = np.nan_to_num(nuts, nan=0.0)
        idx += batch_n
        print(f"Wrote {idx} rows to memmaps")

    return


def run_incremental_pca_on_memmap(emb_path, n_rows, embed_dim, n_components, chunk_size=16384):
    # Re-open read-only memmap for embeddings
    emb_mem = np.lib.format.open_memmap(emb_path, mode="r", dtype="float32", shape=(n_rows, embed_dim))
    ipca = IncrementalPCA(n_components=n_components)
    # partial fit in chunks
    for i in range(0, n_rows, chunk_size):
        chunk = emb_mem[i: i + chunk_size]
        ipca.partial_fit(chunk)
        print(f"IPCA partial_fit on rows {i}..{i+len(chunk)}")

    # Create reduced memmap and transform in chunks
    reduced_path = emb_path.replace("embeddings.npy", "emb_reduced.npy")
    reduced_mem = np.lib.format.open_memmap(reduced_path, mode="w+", dtype="float32", shape=(n_rows, n_components))
    for i in range(0, n_rows, chunk_size):
        chunk = emb_mem[i: i + chunk_size]
        reduced_mem[i: i + len(chunk)] = ipca.transform(chunk)
        print(f"IPCA transform wrote rows {i}..{i+len(chunk)}")

    return reduced_path, n_components


def scale_nutrients_memmap(nut_path, n_rows, n_nutrients, chunk_size=16384):
    nut_mem = np.lib.format.open_memmap(nut_path, mode="r", dtype="float32", shape=(n_rows, n_nutrients))
    scaler = StandardScaler()
    # partial fit
    for i in range(0, n_rows, chunk_size):
        scaler.partial_fit(nut_mem[i: i + chunk_size])
        print(f"StandardScaler partial_fit on rows {i}..{i+chunk_size}")

    scaled_path = nut_path.replace("nutrients.npy", "nutrients_scaled.npy")
    scaled_mem = np.lib.format.open_memmap(scaled_path, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))
    for i in range(0, n_rows, chunk_size):
        scaled_mem[i: i + chunk_size] = scaler.transform(nut_mem[i: i + chunk_size])
        print(f"Wrote scaled nutrients rows {i}..{i+chunk_size}")

    return scaled_path


def run_clustering(args, feature_path, n_rows, n_features, min_cluster_size=5, min_samples=None):
    feat_mem = np.lib.format.open_memmap(feature_path, mode="r", dtype="float32", shape=(n_rows, n_features))
    # load into memory (this should be much smaller)
    X = np.array(feat_mem)  # copy to memory

    model = Birch(threshold=1.0, n_clusters=None)

    # 2. Define the batch size

    # 3. Iterate through the data in batches and use partial_fit
    for i in tqdm(range(0, n_rows, args.batch_size), desc="BIRCH Clustering Progress"):
        # Select a chunk of data
        X_chunk = X[i:i + args.batch_size]

        # Process the chunk incrementally
        model.partial_fit(X_chunk)
    print("Birch done fitting. Predicting...")
    labels = model.predict(X)
    print("Birch scan done.")
    return labels


def write_clusters_to_duckdb(input_db, input_table, id_col, output_db, out_table_clusters, labels_memmap_path):
    # Read labels memmap
    labels_mem = np.load(labels_memmap_path, mmap_mode="r")
    n = labels_mem.shape[0]
    # We stream ids from input table (same order as earlier) and attach labels and insert into output DB
    con_in = duckdb.connect(input_db)
    cur = con_in.cursor()
    cur.execute(f"SELECT {id_col} FROM {input_table}")

    # Prepare output DB and table
    con_out = duckdb.connect(output_db)
    con_out.execute(f"CREATE OR REPLACE TABLE {out_table_clusters} (fdc_id BIGINT, cluster_id INTEGER)")

    idx = 0
    batch = []
    BATCH = 2000
    while True:
        rows = cur.fetchmany(BATCH)
        if not rows:
            break
        for r in rows:
            fdc_id = r[0]
            cluster = int(labels_mem[idx])
            batch.append((fdc_id, cluster))
            idx += 1
        df = pd.DataFrame(batch, columns=["fdc_id", "cluster_id"])
        con_out.register("tmp_insert", df)
        con_out.execute(f"INSERT INTO {out_table_clusters} SELECT * FROM tmp_insert")
        batch = []
        print(f"Inserted {idx} cluster rows")

    con_in.close()
    con_out.close()
    return


def aggregate_nutrients_by_cluster(output_db, clusters_table, input_db, input_table, id_col, nutrient_cols, out_summary_table):
    # We'll attach the input DB inside an output DuckDB connection and run SQL to aggregate
    con = duckdb.connect(output_db)
    # attach/alias the input DB so we can join
    con.execute(f"ATTACH DATABASE '{input_db}' AS src")
    # create summary using SQL join

    nuts = ", ".join([f"AVG(\"{c}\") as \"avg_{c}\"" for c in nutrient_cols])

    con.execute(f'''CREATE OR REPLACE TABLE {out_summary_table} AS
        SELECT c.cluster_id, max(src.{input_table}.food_name) as food_name, {nuts} FROM {clusters_table} c
            INNER JOIN src.{input_table} ON c.fdc_id = src.{input_table}.{id_col}
    GROUP BY c.cluster_id''')

    con.execute(f"COPY {out_summary_table} TO '{out_summary_table}.csv' (HEADER, DELIMITER ',');")
    con.close()
    print(f"Aggregate table {out_summary_table} created in {output_db}")

def final_table(output_db, clusters_table, input_db, input_table, id_col, nutrient_cols, out_summary_table):
    # We'll attach the input DB inside an output DuckDB connection and run SQL to aggregate
    con = duckdb.connect(output_db)
    # attach/alias the input DB so we can join
    con.execute(f"ATTACH DATABASE '{input_db}' AS src")
    # create summary using SQL join

    nuts = ", ".join([f"\"{c}\" as \"{c}\"" for c in nutrient_cols])

    con.execute(f'''CREATE OR REPLACE TABLE {out_summary_table} AS
        SELECT c.cluster_id as cluster_id, c.fdc_id as fdc_id, food_name, {nuts} FROM {clusters_table} c
            INNER JOIN src.{input_table} ON c.fdc_id = src.{input_table}.{id_col} ORDER BY cluster_id ASC''')

    con.execute(f"COPY {out_summary_table} TO '{out_summary_table}.csv' (HEADER, DELIMITER ',');")
    con.close()
    print(f"Aggregate table {out_summary_table} created in {output_db}")

def main():
    args = get_args()
    # Connect and count rows
    con = duckdb.connect(args.input_db)
    count_sql = f"SELECT COUNT(*) FROM {args.input_table}"
    total = con.execute(count_sql).fetchone()[0]
    print(f"Total rows in {args.input_table}: {total}")
    if total == 0:
        print("No rows found. Exiting.")
        return

    # Prepare workdir
    work_dir = Path("fdc_hdbscan")
    print(f"Working directory (memmaps) : {work_dir}")

    # Prepare memmaps
    embed_dim = args.embed_dim
    nutrients = con.execute('SELECT * FROM chosen_nutrients ORDER BY c DESC LIMIT 17').df()

    n_nutrients = len(nutrients)

    reduced_dim = args.pca_components

    feat_path = os.path.join(str(work_dir), "features.npy")

    labels_path = os.path.join(str(work_dir), "labels.npy")

    #mems = prepare_memmaps(total, embed_dim, n_nutrients, str(work_dir))

    # 1) Stream rows and fill raw embeddings + nutrient memmaps
    # write_rows_to_memmaps(con, args.input_table, "fdc_id", "food_name", list(nutrients['nutrient_name']),
    #                       mems["embeddings"], mems["nutrients"],
    #                       args.batch_size)

    # con.close()

    # 2) Run incremental PCA to reduce embedding dimension
    #reduced_path, _ = run_incremental_pca_on_memmap(mems["emb_path"], total, embed_dim, args.pca_components)

    # 3) Scale nutrients
    #scaled_nut_path = scale_nutrients_memmap(mems["nut_path"], total, n_nutrients)

    # 4) Concatenate reduced embeddings + scaled nutrients into feature memmap

    # feat_mem = np.lib.format.open_memmap(feat_path, mode="w+", dtype="float32", shape=(total, reduced_dim + n_nutrients))
    # emb_red_mem = np.lib.format.open_memmap(reduced_path, mode="r", dtype="float32", shape=(total, reduced_dim))
    # scaled_nut_mem = np.lib.format.open_memmap(scaled_nut_path, mode="r", dtype="float32", shape=(total, n_nutrients))
    # for i in range(0, total, 16384):
    #     end = min(total, i + 16384)
    #     feat_mem[i:end, :reduced_dim] = emb_red_mem[i:end]
    #     feat_mem[i:end, reduced_dim:] = scaled_nut_mem[i:end]
    #     print(f"Wrote features rows {i}..{end}")

    # 5) Run Birch scan on the feature memmap
    #labels = run_clustering(args, feat_path, total, reduced_dim + n_nutrients, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)

    # 6) Save labels into memmap (labels memmap was created in prepare_memmaps)

    #np.save(labels_path, labels)

    # 7) Write cluster -> fdc_id mapping to output DB
    write_clusters_to_duckdb(args.input_db, args.input_table, "fdc_id", args.output_db, "clusters", labels_path)

    # 8) Aggregate nutrients by cluster into output DB
    #aggregate_nutrients_by_cluster(args.output_db, "clusters", args.input_db, args.input_table,
        #"fdc_id", nutrients['nutrient_name'], "cluster_nutrient_summary")

    #final_table(args.output_db, "clusters", args.input_db, args.input_table,
    #    "fdc_id", nutrients['nutrient_name'], "output_table")
    print("Done. Output DB contains tables: clusters, cluster_nutrient_summary")
    print(f"Work files retained in {work_dir} (delete when not needed)")


if __name__ == "__main__":
    main()
