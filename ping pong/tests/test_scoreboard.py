"""Scoreboard with a fake MQTT client: publishes float strings, only on change."""

from scoreboard import Scoreboard


class FakeClient:
    def __init__(self):
        self.sent = []
        self.closed = False

    def publish(self, topic, message):
        self.sent.append((topic, message))

    def disconnect(self):
        self.closed = True


def board():
    client = FakeClient()
    return Scoreboard(topic="ME193/Rogers/Jude", client=client), client


def test_first_update_publishes_even_zero():
    b, c = board()
    assert b.update(0)
    assert c.sent == [("ME193/Rogers/Jude", "0.0")]


def test_publishes_float_strings_on_each_change():
    b, c = board()
    for streak in [0, 1, 2, 3, 0]:
        b.update(streak)
    assert [m for _, m in c.sent] == ["0.0", "1.0", "2.0", "3.0", "0.0"]


def test_no_publish_when_streak_unchanged():
    b, c = board()
    for streak in [0, 0, 1, 1, 1, 0, 0]:
        b.update(streak)
    assert [m for _, m in c.sent] == ["0.0", "1.0", "0.0"]


def test_offline_board_does_nothing():
    b = Scoreboard(client=None)
    assert b.update(5) is False


def test_close_disconnects():
    b, c = board()
    b.close()
    assert c.closed
    assert b.update(1) is False
