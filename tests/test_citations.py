import pytest

from mentor.services.answer import is_refusal, parse_citations


@pytest.mark.parametrize(
    ("text", "n", "valid", "invalid"),
    [
        ("Классы объединяют данные [1].", 3, [1], []),
        ("Да [1][3], и ещё раз [1].", 3, [1, 3], []),
        ("Вот так [1, 2].", 3, [1, 2], []),
        ("Вот так [2,3] и [1 , 2].", 3, [2, 3, 1], []),
        ("Ссылка [7] вне диапазона, [0] тоже, а [2] нормальная.", 6, [2], [7, 0]),
        ("Без ссылок.", 6, [], []),
        ("Список [a] и [ 1 ] не считаются.", 6, [], []),
    ],
)
def test_parse_citations(text, n, valid, invalid):
    assert parse_citations(text, n) == (valid, invalid)


@pytest.mark.parametrize(
    "text",
    [
        "В материалах курса этого нет.",
        "в материалах курса этого нет",
        "«В материалах курса этого нет!»",
        "В материалах курса — этого нет...",
        "К сожалению, В МАТЕРИАЛАХ КУРСА ЭТОГО НЕТ.",
        "В материалах курса этого   нет.",
    ],
)
def test_refusal_detected(text):
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Классы предоставляют средство для объединения данных [1].",
        "В материалах курса это есть [2].",
        "",
    ],
)
def test_not_refusal(text):
    assert not is_refusal(text)
