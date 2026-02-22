import asyncio
from pydantic import BaseModel
from kavalai.llm_clients.llm_client import LLMClient
from kavalai.llm_clients.common import StreamContent, Streamer


# Define a Pydantic model for structured output
class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float


async def basic_text_example():
    """
    Demonstrates a simple text-to-text chat completion.
    """
    print("\n--- Basic Text Completion ---")
    client = LLMClient(model="openai/gpt-4o")
    messages = [
        {
            "role": "user",
            "content": "Explain what a Large Language Model is in one sentence.",
        }
    ]

    # chat_completions returns a tuple of (result, stats)
    # Since no response_model is provided, the result is a string.
    result, stats = await client.chat_completions(messages=messages)

    print(f"Result: {result}")
    print(
        f"Stats: {stats.total_tokens} tokens, Cost: ${stats.cost:.4f}, Duration: {stats.duration_seconds:.2f}s"
    )


async def structured_output_example():
    """
    Demonstrates how to get structured output using a Pydantic model.
    The LLM will guarantee the output matches the schema.
    """
    print("\n--- Structured Output (Pydantic) ---")
    # Gemini and OpenAI both support structured output.
    client = LLMClient(model="gemini/gemini-2.0-flash")
    messages = [
        {
            "role": "user",
            "content": "Analyze the following text: 'I absolutely love the new features in Kaval.AI! It's so powerful.'",
        }
    ]

    # Passing response_model makes the result an instance of that model.
    result, stats = await client.chat_completions(
        messages=messages, response_model=Analysis
    )

    print(f"Analysis: {result}")
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")


async def streaming_example():
    """
    Demonstrates how to stream responses for real-time output.
    Uses an asyncio.Queue to receive chunks.
    """
    print("\n--- Streaming Response ---")
    client = LLMClient(model="openai/gpt-4o")
    messages = [{"role": "user", "content": "Write 4 line poem about coding."}]

    queue = asyncio.Queue()
    streamer = Streamer("response_text", queue)

    # Run chat_completions in a task to consume the queue concurrently
    task = asyncio.create_task(
        client.chat_completions(messages=messages, streamer=streamer)
    )

    print("Streaming: ", end="", flush=True)
    while True:
        chunk = StreamContent.model_validate_json(await queue.get())
        if chunk.type == "partial":
            print(f"{chunk.value}", end="", flush=True)
        if chunk.type == "complete":
            print(f"\nComplete message received: {chunk.value}")
            break

    result, stats = await task
    print(f"\nFinal Stats: {stats.total_tokens} tokens, Cost: ${stats.cost:.4f}")


async def multimodal_image_example():
    """
    Demonstrates sending an image to an LLM (Multimodal).
    Images are typically passed as base64 strings in the message content.
    """
    print("\n--- Multimodal (Image Input) ---")
    # For this example, we'll create a dummy base64 pixel if a real image isn't provided.
    # In a real app, you'd read a file: base64.b64encode(open("image.jpg", "rb").read()).decode("utf-8")
    dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    client = LLMClient(model="openai/gpt-4o")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{dummy_image_b64}",
                },
            ],
        }
    ]

    try:
        result, stats = await client.chat_completions(messages=messages)
        print(f"Image Analysis: {result}")
    except Exception as e:
        print(
            f"Note: This might fail if the dummy image is too small or invalid for the provider: {e}"
        )


async def reasoning_and_thinking_example():
    """
    Demonstrates reasoning/thinking models (like Gemini 2.0 Flash Thinking or OpenAI o1/o3).
    These models show their "thoughts" before the final answer.
    """
    print("\n--- Reasoning and Thinking (Gemini Example) ---")
    # Gemini 2.0 Flash Thinking supports reasoning parameters.
    client = LLMClient(model="gemini/gemini-3-flash-preview")
    messages = [
        {
            "role": "user",
            "content": "If I have 3 apples and you give me 2 more, but then I eat one, how many do I have? Explain step by step.",
        }
    ]

    queue = asyncio.Queue()
    streamer = Streamer("streamed_output", queue)
    # thinking_budget is a Gemini-specific parameter (in seconds)
    task = asyncio.create_task(
        client.chat_completions(
            messages=messages,
            streamer=streamer,
            thinking_budget=10,  # Use up to 10 seconds for thinking
        )
    )

    print("Thinking/Reasoning: ", end="", flush=True)
    has_answer_started = False
    while True:
        chunk = StreamContent.model_validate_json(await queue.get())
        if chunk.type == "partial":
            if chunk.name.endswith("_thought"):
                print(f"{chunk.value}", end="", flush=True)
            else:
                if not has_answer_started:
                    print("\nAnswer: ", end="", flush=True)
                    has_answer_started = True
                print(f"{chunk.value}", end="", flush=True)
        if chunk.type == "complete":
            print(f"\nComplete message received: {chunk.value}")
            break

    result, stats = await task
    print(f"\nFinal Stats: {stats.total_tokens} tokens")


async def openai_reasoning_example():
    """
    Demonstrates OpenAI o1-preview or o3-mini reasoning.
    Uses 'reasoning_effort' parameter.
    """
    print("\n--- OpenAI Reasoning (o3-mini) ---")
    client = LLMClient(model="openai/o3-mini")
    messages = [{"role": "user", "content": "Solve for x: 2x + 5 = 15"}]

    # reasoning_effort can be 'low', 'medium', or 'high'
    result, stats = await client.chat_completions(
        messages=messages, reasoning={"effort": "low"}
    )
    print(f"Result: {result}")


async def main():
    # Set your API keys in environment variables:
    # os.environ["OPENAI_API_KEY"] = "..."
    # os.environ["GEMINI_API_KEY"] = "..."

    await basic_text_example()
    await structured_output_example()
    await streaming_example()
    await multimodal_image_example()
    await reasoning_and_thinking_example()
    await openai_reasoning_example()


if __name__ == "__main__":
    asyncio.run(main())
