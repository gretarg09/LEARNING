from logic.adding import adding_two_numbers

def test_adding_two_numbers():
    a = 2
    b = 5

    result = adding_two_numbers(a, b)

    assert result == 7
