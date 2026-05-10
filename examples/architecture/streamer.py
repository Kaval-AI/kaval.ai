import asyncio

from kavalai import Streamer


async def stream_messages_example() -> None:
    streamer = Streamer()
    message_streamer = streamer.get_value_streamer("result")
    await message_streamer.stream_partial("Hello,")
    await message_streamer.stream_partial(" world!")
    await message_streamer.stream_complete()  # Marks message "result" complete

    async for message in streamer:
        print(message.model_dump_json())


print("stream_messages_example()")
asyncio.run(stream_messages_example())
print(""" Should print:
{"type":"partial","name":"result","value":"Hello,"}
{"type":"partial","name":"result","value":"Hello, world!"}
{"type":"complete","name":"result","value":"Hello, world!"}
""")


async def stream_messages_background_task_example() -> None:
    streamer = Streamer()
    message_streamer = streamer.get_value_streamer("result")

    # Run streaming in background task
    async def stream_messages() -> None:
        await message_streamer.stream_partial("Hello,")
        await message_streamer.stream_partial(" world!")
        await message_streamer.stream_complete()  # Marks message "result" complete

    # Start streaming task
    streaming_task = asyncio.create_task(stream_messages())

    # Consume messages
    async for message in streamer:
        print(message.model_dump_json())

    # Wait for streaming to complete
    await streaming_task


print("stream_messages_background_task_example()")
asyncio.run(stream_messages_background_task_example())
print("""Should print:
{"type":"partial","name":"result","value":"Hello,"}
{"type":"partial","name":"result","value":"Hello, world!"}
{"type":"complete","name":"result","value":"Hello, world!"}
""")


async def stream_messages_delta_example() -> None:
    streamer = Streamer(stream_delta=True)
    message_streamer = streamer.get_value_streamer("result")
    await message_streamer.stream_partial("Hello,")
    await message_streamer.stream_partial(" world!")
    await message_streamer.stream_complete()  # Marks message "result" complete

    async for message in streamer:
        print(message.model_dump_json())


print("stream_messages_delta_example()")
asyncio.run(stream_messages_delta_example())
print(""" Should print:
{"type":"partial","name":"result","value":"Hello,"}
{"type":"partial","name":"result","value":" world!"}
{"type":"complete","name":"result","value":null}
""")


from pydantic import BaseModel


class PersonRecord(BaseModel):
    name: str
    birth_year: int


async def stream_pydantic_type_example() -> None:
    streamer = Streamer()
    message_streamer = streamer.get_value_streamer(
        "result", response_model=PersonRecord
    )
    await message_streamer.stream_partial('{"name": "Ti')
    await message_streamer.stream_partial('mo", ')
    await message_streamer.stream_partial(' "birth_year": 1986}')
    await message_streamer.stream_complete()

    async for message in streamer:
        print(message.model_dump_json())
        print("Partial JSON: ", message.value)


print("stream_pydantic_type_example()")
asyncio.run(stream_pydantic_type_example())
print(""" Should print:
{"type":"partial","name":"result","value":"Hello,"}
{"type":"partial","name":"result","value":" world!"}
{"type":"complete","name":"result","value":null}
""")
