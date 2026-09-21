from generation.prompt import RAG_PROMPT, GENERAL_PROMPT


def test_rag_prompt_contains_context_and_question():
    messages = RAG_PROMPT.invoke(
        {
            "context": "Refunds are allowed within 30 days.",
            "question": "What is the refund policy?",
        }
    )

    prompt_text = str(messages)

    assert "Refunds are allowed within 30 days." in prompt_text
    assert "What is the refund policy?" in prompt_text


def test_general_prompt_contains_question():
    messages = GENERAL_PROMPT.invoke(
        {
            "question": "Hello, how are you?"
        }
    )

    prompt_text = str(messages)

    assert "Hello, how are you?" in prompt_text