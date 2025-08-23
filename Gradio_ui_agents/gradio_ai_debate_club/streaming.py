import time
from typing import Iterator
from config import CHUNK_DELAY, BATCH_CHARS

def stream_llm(llm, messages) -> Iterator[str]:
    """
    Stream content from a language model in batches.

    This function aggregates small content pieces into batches and yields
    growing text chunks. It uses a buffer to accumulate content until a
    specified batch size is reached, then yields the accumulated text.

    Args:
        llm: The language model instance to stream content from.
        messages: The input messages to send to the language model.

    Yields:
        str: Growing text chunks streamed from the language model.

    Raises:
        Exception: If streaming fails, falls back to invoking the model directly.
    """
    acc = ""
    buffer = ""
    try:
        for chunk in llm.stream(messages):
            piece = getattr(chunk, "content", "") or ""
            
            if not piece:
                continue
            
            buffer += piece
            if len(buffer) >= BATCH_CHARS:
                acc += buffer
                buffer = ""
                yield acc
                time.sleep(CHUNK_DELAY)
                
        if buffer:
            acc += buffer
            yield acc
            
    except Exception:
        resp = llm.invoke(messages)
        yield resp.content
