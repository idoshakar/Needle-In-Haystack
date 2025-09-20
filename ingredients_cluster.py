import argparse
import os
from tqdm import tqdm
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

import torch
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import re

def get_args():
    '''
    Parses the arguments and returns them.

    Returns: The parsed arguments.
    '''
    p = argparse.ArgumentParser()
    p.add_argument("--input-db", default="food_central.db", help="Input DuckDB file path")
    p.add_argument("--input-table", default="pivoted_full_join", help="Table inside input DB containing food rows")
    p.add_argument("--output-db", default="foods.db", help="Output DuckDB file path to write clusters & summaries")
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--nutrients-scale", type=float, default=4.0)
    p.add_argument("--embed-dim", type=int, default=EMBED_SIZE, help="Expected embedding dimension of model")
    p.add_argument("--batch-size", type=int, default=1024, help="DB fetch batch size")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=None)
    args = p.parse_args()
    args.suffix = f'{args.threshold}_bi_kmeans_{args.nutrients_scale}_nut'
    args.output_db += f"_{args.suffix}"
    return args

class Paths(object):
    '''
    See get_paths.
    '''
    def __init__(self, args, work_dir: Path):
        self.feat = os.path.join(str(work_dir), "features.npy")

        self.labels = os.path.join(str(work_dir), f"labels_{args.suffix}.npy")

        self.emb = os.path.join(work_dir, "embeddings.npy")
        self.nut = os.path.join(work_dir, "nutrients.npy")
        self.reduced = os.path.join(work_dir, "emb_reduced.npy")
        self.scaled_nut = os.path.join(work_dir, "nutrients_scaled.npy")
        self.labels = os.path.join(work_dir, "labels.npy")

        self.emb_reduced = self.emb.replace("embeddings.npy", "emb_reduced.npy")
        self.scaled_nut = self.nut.replace("nutrients.npy", "nutrients_scaled.npy")

def get_paths(args, work_dir: Path) -> Paths:
    '''
    Receives the arguments and work directory, and returns the paths
    that will be used.

    args: Command line arguments.
    work_dir: The work directory.

    Returns: The paths that will be used.
    '''
    return Paths(args, work_dir)


def prepare_memmaps(n_rows, embed_dim, n_nutrients, paths):
    '''
    Create memory map files for embeddings, nutrients, reduced embeddings, scaled nutrients, and labels.

    n_rows: The number of rows.
    embed_dim: The sentence embedding dimension - post PCA.
    n_nutrients: The number of nutrients.
    paths: The memory map paths to be used.
    '''
    embeddings = np.lib.format.open_memmap(paths.emb, mode="w+", dtype="float32", shape=(n_rows, embed_dim))
    nutrients = np.lib.format.open_memmap(paths.nut, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))

    scaled_nuts = np.lib.format.open_memmap(paths.scaled_nut, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))
    labels = np.lib.format.open_memmap(paths.labels, mode="w+", dtype="int32", shape=(n_rows,))


    return {
        "embeddings": embeddings,
        "nutrients": nutrients,
        "scaled_nuts": scaled_nuts,
        "labels": labels
    }

MAX_LENGTH = 20
EMBED_SIZE = 100

def clean_text(text: str) -> str:
    '''
    Cleans the text for future encoding.

    text: The text string.
    Returns: The cleaned text.
    '''
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(s: str) -> list[str]:
    '''
    Transforms a sentence into a list of words

    s: The sentence.
    Returns: A list of processed words.
    '''
    s = clean_text(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]

def encode_glove(s: str) -> pd.Series:
    '''
    Encodes a sentence using GloVe encoding.

    s: The sentence.
    Returns: A vector with the encoded sentence.
    '''
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

def bisecting_kmeans(X: np.ndarray, args) -> list[list[int]]:
    '''
    Performs bisecting Mini Batch K-Means on the given data.

    X: The data.
    args: Command line arguments.

    Returns: A list of clusters. Each cluster is a list of row indices of the original data.
    '''
    clusters = [np.arange(len(X))]
    final_clusters = []

    pbar = tqdm(desc="Bisecting K-Means")

    while clusters:
        idx = clusters.pop()
        sub_X = X[idx]


        cluster_variance = np.var(sub_X, axis=0).sum()
        if cluster_variance <= args.threshold:
            final_clusters.append(idx)
            pbar.update(1)
            continue


        kmeans = MiniBatchKMeans(n_clusters=2, batch_size=args.batch_size)
        if 0 not in sub_X.shape:
            labels = kmeans.fit_predict(sub_X)

            for lbl in [0, 1]:
                sub_idx = idx[labels == lbl]
                clusters.append(sub_idx)

    pbar.close()
    return final_clusters

def write_rows_to_memmaps(con, table, id_col, text_col, nutrient_cols, embeddings_memmap, nutrients_memmap,
                          fetch_batch):

    '''
    Writes the encoded table rows to the given memory maps.
    Each nutrient has its own column.

    con: The database connection.
    table: The name of the table in the database containing the data.
    id_col: The id column name.
    text_col: The food name column.
    nutrient_cols: The nutrient column names.
    embeddings_memmap: The memory map of the food embeddings.
    nutrients_memmap: The memory map of the food nutrients.
    fetch_batch: Size of each batch to fetch. Controls memory usage.
    '''
    cursor = con.cursor()
    sql = f"SELECT {id_col}, {text_col}, \"{'", "' .join(nutrient_cols)}\" FROM {table}"
    cursor.execute(sql)

    idx = 0
    while True:
        rows = cursor.fetchmany(fetch_batch)
        if not rows:
            break

        texts = [r[1] if r[1] is not None else "" for r in rows]
        nuts = np.array([[None if v is None else float(v) for v in r[2:]] for r in rows], dtype="float32")

        emb = []
        for text in texts:
            emb.append(encode_glove(text))

        embeddings_memmap[idx: idx + len(emb), :] = emb
        nutrients_memmap[idx: idx + len(emb), :] = np.nan_to_num(nuts, nan=0.0)
        idx += len(emb)
        print(f"Wrote {idx} rows to memmaps")

    return


def run_incremental_pca_on_memmap(paths: Paths, n_rows: int, embed_dim: int, n_components: int, chunk_size=16384):
    '''
    Runs incremental PCA on the food embeddings memory map and writes the result
    to a memory map.

    paths: The paths for the memory maps.
    n_rows: The number of rows.
    n_components: The number of components.
    chunk_size: The size of the chunk to be used in each incremental PCA step.
    '''
    emb_mem = np.lib.format.open_memmap(paths.emb, mode="r", dtype="float32", shape=(n_rows, embed_dim))
    ipca = IncrementalPCA(n_components=n_components)

    for i in range(0, n_rows, chunk_size):
        chunk = emb_mem[i: i + chunk_size]
        ipca.partial_fit(chunk)
        print(f"IPCA partial_fit on rows {i}..{i+len(chunk)}")


    reduced_mem = np.lib.format.open_memmap(paths.emb_reduced, mode="w+", dtype="float32", shape=(n_rows, n_components))
    for i in range(0, n_rows, chunk_size):
        chunk = emb_mem[i: i + chunk_size]
        reduced_mem[i: i + len(chunk)] = ipca.transform(chunk)
        print(f"IPCA transform wrote rows {i}..{i+len(chunk)}")


def scale_nutrients_memmap(paths: Paths, n_rows: int, n_nutrients: int, chunk_size=16384):
    '''
    Performs standard scaling on the nutrients memory map, and writes the result
    to a new memory map.

    n_rows: The number of rows.
    n_nutrients: The number of nutrients.
    chunk_size: The size of the chunk to be used in each StandardScaler partial fit.
    '''
    nut_mem = np.lib.format.open_memmap(paths.nut, mode="r", dtype="float32", shape=(n_rows, n_nutrients))
    scaler = StandardScaler()

    for i in range(0, n_rows, chunk_size):
        scaler.partial_fit(nut_mem[i: i + chunk_size])
        print(f"StandardScaler partial_fit on rows {i}..{i+chunk_size}")

    scaled_mem = np.lib.format.open_memmap(paths.scaled_nut, mode="w+", dtype="float32", shape=(n_rows, n_nutrients))
    for i in range(0, n_rows, chunk_size):
        scaled_mem[i: i + chunk_size] = scaler.transform(nut_mem[i: i + chunk_size])
        print(f"Wrote scaled nutrients rows {i}..{i+chunk_size}")


def run_clustering(args, feature_path: str, n_rows: int, n_features: int, n_nutrients: int):
    '''
    Runs clustering on the features memory map.

    args: Command line arguments.
    n_rows: The number of rows.
    n_features: The number of features.
    n_nutrients: The number of nutrients.

    Returns: The cluster for each row.
    '''
    feat_mem = np.lib.format.open_memmap(feature_path, mode="r", dtype="float32", shape=(n_rows, n_features))

    X = np.array(feat_mem)
    X = X[:, n_nutrients:] * args.nutrients_scale

    clusters = bisecting_kmeans(X, args)

    labels = np.empty(X.shape[0], dtype=int)
    for cluster_id, cluster_indices in enumerate(clusters):
        labels[cluster_indices] = cluster_id

    print("bisecting_kmeans done.")
    return labels


def write_clusters_to_duckdb(input_db, input_table, id_col, output_db, out_table_clusters, labels_memmap_path):
    '''
    Creates a new table with each row id and its corresponding cluster.

    input_db: The input database.
    input_table: The table containing the ingredients.
    id_col: The food id column.
    output_db: The output database.
    output_table_clusters: A table with each row id and its corresponding cluster id.
    labels_memmap_path: The memory map path of the ingredient cluster labels.
    '''
    labels_mem = np.load(labels_memmap_path, mmap_mode="r")

    con_in = duckdb.connect(input_db)
    cur = con_in.cursor()
    cur.execute(f"SELECT {id_col} FROM {input_table}")


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

def final_table(output_db, clusters_table, input_db, input_table, id_col, nutrient_cols, out_summary_table):
    '''
    Outputs the final table.
    Format:
    cluster_id, fdc_id, food_name, nutrient_cols...

    output_db: The output database.
    clusters_table: The cluster_id - food id table.
    input_table: The food table.
    id_col: The food id column name.
    nutrient_cols: The nutrient columns' names.
    out_summary_table: The final table's name.
    '''
    con = duckdb.connect(output_db)

    con.execute(f"ATTACH DATABASE '{input_db}' AS src")

    nuts = ", ".join([f"\"{c}\" as \"{c}\"" for c in nutrient_cols])

    con.execute(f'''CREATE OR REPLACE TABLE "{out_summary_table}" AS
        SELECT c.cluster_id as cluster_id, c.fdc_id as fdc_id, food_name, {nuts} FROM {clusters_table} c
            INNER JOIN src.{input_table} ON c.fdc_id = src.{input_table}.{id_col} ORDER BY cluster_id ASC''')

    con.execute(f"COPY \"{out_summary_table}\" TO '{out_summary_table}.csv' (HEADER, DELIMITER ',');")
    con.close()
    print(f"Final table {out_summary_table} created in {output_db}")

def final_table_sample(output_db, clusters_table, input_db, input_table, id_col, nutrient_cols, out_summary_table):
    '''
    Outputs a sample from the final table - for debugging purposes
    Format:
    cluster_id, fdc_id, food_name, nutrient_cols...

    output_db: The output database.
    clusters_table: The cluster_id - food id table.
    input_table: The food table.
    id_col: The food id column name.
    nutrient_cols: The nutrient columns' names.
    out_summary_table: The final table's name.
    '''
    con = duckdb.connect(output_db)
    out_summary_table += '_sample'

    con.execute(f"ATTACH DATABASE '{input_db}' AS src")

    nuts = ", ".join([f"\"{c}\" as \"{c}\"" for c in nutrient_cols])

    con.execute(f'''
        CREATE OR REPLACE TABLE "{out_summary_table}" AS
            SELECT c.cluster_id as cluster_id, c.fdc_id as fdc_id, food_name, {nuts} FROM {clusters_table} c
                    INNER JOIN src.{input_table} ON c.fdc_id = src.{input_table}.{id_col}
                WHERE c.cluster_id >= 70000 and c.cluster_id <=180000
        ORDER BY cluster_id ASC''')

    con.execute(f"COPY \"{out_summary_table}\" TO '{out_summary_table}.csv' (HEADER, DELIMITER ',');")
    con.close()
    print(f"Final table {out_summary_table} created in {output_db}")

def main():
    args = get_args()

    con = duckdb.connect(args.input_db)
    count_sql = f"SELECT COUNT(*) FROM {args.input_table}"
    total = con.execute(count_sql).fetchone()[0]

    print(f"Total rows in {args.input_table}: {total}")
    if total == 0:
        print("No rows found. Exiting.")
        return

    work_dir = Path("fdc_work")
    print(f"Working directory (memmaps) : {work_dir}")

    nutrients = con.execute('SELECT * FROM chosen_nutrients ORDER BY c DESC LIMIT 17').df()

    n_nutrients = len(nutrients)

    reduced_dim = args.pca_components

    paths = get_paths(args, work_dir)

    mems = prepare_memmaps(total, args.embed_dim, n_nutrients, paths)

    write_rows_to_memmaps(con, args.input_table, "fdc_id", "food_name", list(nutrients['nutrient_name']),
                          mems["embeddings"], mems["nutrients"],
                          args.batch_size)

    con.close()

    run_incremental_pca_on_memmap(paths, total, args.embed_dim, args.pca_components)

    scale_nutrients_memmap(paths, total, n_nutrients)

    feat_mem = np.lib.format.open_memmap(paths.feat, mode="w+", dtype="float32", shape=(total, reduced_dim + n_nutrients))
    emb_red_mem = np.lib.format.open_memmap(paths.emb_reduced, mode="r", dtype="float32", shape=(total, reduced_dim))
    scaled_nut_mem = np.lib.format.open_memmap(paths.scaled_nut, mode="r", dtype="float32", shape=(total, n_nutrients))
    for i in range(0, total, 16384):
        end = min(total, i + 16384)
        feat_mem[i:end, :reduced_dim] = emb_red_mem[i:end]
        feat_mem[i:end, reduced_dim:] = np.array(scaled_nut_mem[i:end, :n_nutrients])
        print(f"Wrote features rows {i}..{end}")

    labels = run_clustering(args, paths.feat, total, reduced_dim + n_nutrients, n_nutrients)
    np.save(paths.labels, labels)

    write_clusters_to_duckdb(args.input_db, args.input_table, "fdc_id", args.output_db, "clusters", paths.labels)

    final_table_sample(args.output_db, "clusters", args.input_db, args.input_table,
        "fdc_id", nutrients['nutrient_name'], f"output_table_{args.suffix}")

    print(f"Done. Output DB contains tables: clusters, cluster_nutrient_summary_{args.suffix}, output_table_{args.suffix}")
    print(f"Work files retained in {work_dir} (delete when not needed)")


if __name__ == "__main__":
    main()
