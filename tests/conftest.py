import threading

import pytest


@pytest.fixture
def gate():
    """Evento que segura uma chamada simulada (timeout real sem `sleep`).

    É sempre liberado no teardown: se o teste falhar antes de liberar, a thread presa no
    executor travaria a saída do processo.
    """
    event = threading.Event()
    yield event
    event.set()
