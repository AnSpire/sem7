import pytest

from mentor.adapters.ollama import strip_thinking


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ответ [1].", "Ответ [1]."),
        ("<think>рассуждение</think>\n\nОтвет [1].", "Ответ [1]."),
        ("<think>\nмного\nстрок\n</think>Ответ", "Ответ"),
        (
            "Хорошо, мне нужно ответить... [2] не подходит.\n</think>\n\nОбъект класса [1]",
            "Объект класса [1]",
        ),
        ("<think></think>\nВ материалах курса этого нет.", "В материалах курса этого нет."),
    ],
)
def test_strip_thinking(raw, expected):
    assert strip_thinking(raw) == expected
