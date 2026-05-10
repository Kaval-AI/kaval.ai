Streamer class
==============

Basic usage
-----------

Streamer class is useful for displaying partial responses from the LLM models.
The example below shows streaming value with label `result`.

.. code-block:: python

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

Will print:

.. code-block:: json

    {"type":"partial","name":"result","value":"Hello,"}
    {"type":"partial","name":"result","value":"Hello, world!"}
    {"type":"complete","name":"result","value":"Hello, world!"}


Pydantic class support
----------------------

To simplify processing structured outputs from LLM-s, the `Streamer` supports
Pydantic models.

.. code-block:: python

    import asyncio
    from kavalai import Streamer
    from pydantic import BaseModel

    class PersonRecord(BaseModel):
        name: str
        birth_year: int


    async def stream_pydantic_type_example() -> None:
        streamer = Streamer()
        message_streamer = streamer.get_value_streamer("result", response_model=PersonRecord)
        await message_streamer.stream_partial('{"name": "Ti')
        await message_streamer.stream_partial('mo", ')
        await message_streamer.stream_partial(' "birth_year": 1986}')
        await message_streamer.stream_complete()

        async for message in streamer:
            print(message.model_dump_json())
            print('Partial JSON: ', message.value)

Note that the streamed messages are forced to be valid JSON even if partial value is streamed.

.. code-block:: json

    {"type":"partial","name":"result","value":"{\"name\": \"Ti\"}"}
    {"name": "Ti"}

    {"type":"partial","name":"result","value":"{\"name\": \"Timo\"}"}
    {"name": "Timo"}

    {"type":"partial","name":"result","value":"{\"name\": \"Timo\", \"birth_year\": 1986}"}
    {"name": "Timo", "birth_year": 1986}

    {"type":"complete","name":"result","value":"{\"name\": \"Timo\", \"birth_year\": 1986}"}
    {"name": "Timo", "birth_year": 1986}


Delta streaming
---------------

It is possible to stream individual chunks as they arrive from the source
by setting `stream_delta` parameter `True`.
In this case we don't store the partial chunks in the buffer, but stream them directly to the caller.
Message type `complete` indicates that all the chunks are received and complete
value can be now constructed from the chunks. Of course, the value itself will
be omitted, because we don't have it in the buffer.

.. code-block:: python

    async def stream_messages_delta_example() -> None:
        streamer = Streamer(stream_delta=True)
        message_streamer = streamer.get_value_streamer("result")
        await message_streamer.stream_partial("Hello,")
        await message_streamer.stream_partial(" world!")
        await message_streamer.stream_complete()  # Marks message "result" complete

        async for message in streamer:
            print(message.model_dump_json())

    asyncio.run(stream_messages_delta_example())

Should print

.. code-block:: json

    {"type":"partial","name":"result","value":"Hello,"}
    {"type":"partial","name":"result","value":" world!"}
    {"type":"complete","name":"result","value":null}
