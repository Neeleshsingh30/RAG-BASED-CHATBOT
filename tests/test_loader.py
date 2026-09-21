import pytest

from ingestion.loader import load_documents


def test_load_text_document(tmp_path):
    document = tmp_path / "example.txt"

    document.write_text(
        "This is a test document.",
        encoding="utf-8",
    )

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0].page_content == "This is a test document."
    assert documents[0].metadata["source"]


def test_load_markdown_document(tmp_path):
    document = tmp_path / "example.md"

    document.write_text(
        "# Test Document\n\nThis is markdown content.",
        encoding="utf-8",
    )

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert "Test Document" in documents[0].page_content


def test_load_documents_invalid_directory():
    with pytest.raises(ValueError):
        load_documents("this-directory-does-not-exist")


def test_load_documents_with_no_supported_files(tmp_path):
    unsupported_file = tmp_path / "example.csv"

    unsupported_file.write_text(
        "name,value\nA,10",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_documents(str(tmp_path))