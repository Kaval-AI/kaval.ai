class BaseClient:
    """Base class for LLM clients."""

    def __init__(self):
        pass


class BaseLlmModel(BaseClient):
    def __init__(self):
        super().__init__()


class BaseEmbeddingsModel(BaseClient):
    def __init__(self):
        super().__init__()
