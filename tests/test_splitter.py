from langchain_core.documents import Document

from ingestion.splitter import split_into_chunks


def test_split_into_chunks():
    document = Document(
        page_content="This is a test document. " * 100,
        metadata={"source": "test.txt"},
    )

    chunks = split_into_chunks(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 100


def test_splitter_preserves_metadata():
    document = Document(
        page_content="This is some test content.",
        metadata={"source": "test.txt"},
    )

    chunks = split_into_chunks(
        [document],
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "test.txt"