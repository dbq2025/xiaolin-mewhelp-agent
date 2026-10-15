def test_fastapi_sse_available():
    from fastapi.sse import EventSourceResponse, ServerSentEvent  # noqa: F401


def test_langchain_importable():
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
    from langchain_openai import ChatOpenAI  # noqa: F401

    assert HumanMessage("hi").content == "hi"
