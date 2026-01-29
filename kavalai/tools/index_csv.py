"""
This tool indexes large CSV files into a RAG (Retrieval-Augmented Generation) collection.
It handles batching, metadata field selection, and allows ignoring specific columns.

Example usage:
python -m kavalai.tools.index_csv local_data/song_lyrics.csv \
    --collection-name lyrics \
    --embedding-profile openai \
    --metadata-fields id title artist \
    --index-fields lyrics \
    --source-field id \
    --mode full \
    --replace
    --limit 100

Modes:
- full: (Default) The entire field is indexed as a single RAG entry.
- lines: The field is split into lines and each line is indexed separately.
"""

import asyncio
import argparse
import csv
import os
import sys
from typing import List, Optional, Generator, Dict

from kavalai.agents.db import db_manager
from kavalai.agents.rag_service import RagService
import logging


logger = logging.getLogger(__name__)


def csv_row_generator(
    csv_path: str, limit: Optional[int] = None
) -> Generator[Dict[str, str], None, None]:
    """Generator that yields rows from a CSV file up to a limit."""
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if limit is not None and count >= limit:
                break
            yield row
            count += 1


def content_splitter_generator(
    rows: Generator[Dict[str, str], None, None],
    index_fields: List[str],
    metadata_fields: List[str],
    source_field: str,
    mode: str,
) -> Generator[Dict[str, any], None, None]:
    """Generator that splits row content based on the mode."""
    for row in rows:
        row_meta = {col: val for col, val in row.items() if col in metadata_fields}
        source_id = row.get(source_field, "default")

        for field in index_fields:
            if field not in row:
                continue

            content = row[field]
            if mode == "lines":
                for line in content.splitlines():
                    if line.strip():
                        yield {
                            "text": line.strip(),
                            "meta": row_meta,
                            "source_id": source_id,
                        }
            else:  # full
                yield {"text": content, "meta": row_meta, "source_id": source_id}


async def index_csv(
    csv_path: str,
    collection_name: str,
    embedding_profile_name: str,
    metadata_fields: List[str],
    index_fields: List[str],
    source_field: str,
    mode: str,
    limit: Optional[int],
    replace: bool = False,
    batch_size: int = 10,
):
    async_session = db_manager.get_sessionmaker(uri=os.environ["KAVALAI_DB_URI"])
    async with async_session() as session:
        # Upsert profile to DB to get ID
        rag_service = RagService(session, os.environ["DEFAULT_EMBEDDING_MODEL"])

        rows_processed = 0
        total_chunks = 0

        rows_gen = csv_row_generator(csv_path, limit)

        while True:
            batch_rows = []
            try:
                for _ in range(batch_size):
                    batch_rows.append(next(rows_gen))
            except StopIteration:
                pass

            if not batch_rows:
                break

            texts = []
            metas = []
            source_ids = []

            for entry in content_splitter_generator(
                iter(batch_rows), index_fields, metadata_fields, source_field, mode
            ):
                texts.append(entry["text"])
                metas.append(entry["meta"])
                source_ids.append(entry["source_id"])

            if texts:
                if replace:
                    # Collect unique source_ids in this batch to delete them
                    unique_source_ids = list(set(source_ids))
                    await rag_service.delete_by_source_ids(
                        collection_name, unique_source_ids
                    )

                rows_processed += len(batch_rows)
                total_chunks += len(texts)
                logger.info(
                    f"Indexing batch of {len(batch_rows)} rows generating {len(texts)} chunks (Total rows: {rows_processed}, total chunks: {total_chunks})..."
                )
                await rag_service.batch_index(
                    texts, metas, collection_name=collection_name, source_ids=source_ids
                )
            else:
                rows_processed += len(batch_rows)

    logger.info(
        f"Finished indexing {rows_processed} rows ({total_chunks} chunks) into collection '{collection_name}'."
    )


def main():
    parser = argparse.ArgumentParser(description="Index a CSV file into RAG.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--collection-name", required=True, help="RAG collection name")
    parser.add_argument(
        "--embedding-profile",
        required=True,
        help="Embedding profile name (from embedding_profiles/ folder)",
    )
    parser.add_argument(
        "--metadata-fields", nargs="*", default=[], help="Fields to store in metadata"
    )
    parser.add_argument(
        "--index-fields",
        nargs="+",
        required=True,
        help="Fields to include in indexed content",
    )
    parser.add_argument(
        "--source-field",
        required=True,
        help="Field to store in source_id of the rag_index table",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete matching rows (collection_name, source_id) before updating",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "lines"],
        default="full",
        help="How to index content (full: entire field as one entry, lines: each line as an entry)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of rows to index")

    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for indexing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        logger.error(f"Error: CSV file '{args.csv_path}' not found.")
        sys.exit(1)

    asyncio.run(
        index_csv(
            args.csv_path,
            args.collection_name,
            args.embedding_profile,
            args.metadata_fields,
            args.index_fields,
            args.source_field,
            args.mode,
            args.limit,
            args.replace,
            args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
