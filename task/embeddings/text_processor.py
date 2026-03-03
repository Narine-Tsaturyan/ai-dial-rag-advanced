from enum import StrEnum
import psycopg2
from psycopg2.extras import RealDictCursor
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text

class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"
    COSINE_DISTANCE = "cosine"

class TextProcessor:
    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        return psycopg2.connect(

            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors;")
                conn.commit()

    def _save_chunk(self, document_name, chunk_text, embedding):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cur.execute(
                    """
                    INSERT INTO vectors (document_name, text, embedding)
                    VALUES (%s, %s, %s::vector)
                    """,
                    (document_name, chunk_text, emb_str)
                )
                conn.commit()

    def process_text_file(self, file_path, chunk_size=300, overlap=40, dimensions=1536, truncate_table=True):
        document_name = file_path.split('/')[-1]
        if truncate_table:
            self._truncate_table()
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Indexing {len(chunks)} chunks...")
        embeddings_dict = self.embeddings_client.get_embeddings(chunks)
        for idx, chunk in enumerate(chunks):
            embedding = embeddings_dict[idx]
            if len(embedding) != dimensions:
                raise ValueError(f"Embedding dimension mismatch: expected {dimensions}, got {len(embedding)}")
            self._save_chunk(document_name, chunk, embedding)
        print("Indexing complete.")

    def search(self, query, search_mode=SearchMode.COSINE_DISTANCE, top_k=5, dimensions=1536):
        embeddings_dict = self.embeddings_client.get_embeddings([query])
        query_embedding = embeddings_dict[0]
        if len(query_embedding) != dimensions:
            raise ValueError(f"Embedding dimension mismatch: expected {dimensions}, got {len(query_embedding)}")
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        operator = '<=>' if search_mode == SearchMode.COSINE_DISTANCE else '<->'
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT text, embedding {operator} %s::vector AS score
                    FROM vectors
                    ORDER BY score ASC
                    LIMIT %s
                    """,
                    (emb_str, top_k)
                )
                results = cur.fetchall()
        print(f"Found {len(results)} results from vector search.")
        return [(row['text'], row['score']) for row in results]