from generation.parser import AnswerSchema, Citation


def test_answer_schema_without_citations():
    answer = AnswerSchema(
        answer="This is a test answer."
    )

    assert answer.answer == "This is a test answer."
    assert answer.citations == []


def test_answer_schema_with_citation():
    citation = Citation(
        source="data/raw/example.txt",
        snippet="Example supporting text."
    )

    answer = AnswerSchema(
        answer="This answer is supported by the document.",
        citations=[citation],
    )

    assert answer.answer == "This answer is supported by the document."
    assert len(answer.citations) == 1
    assert answer.citations[0].source == "data/raw/example.txt"
    assert answer.citations[0].snippet == "Example supporting text."