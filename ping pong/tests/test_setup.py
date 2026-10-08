"""Phase 0 smoke test: the copied libraries import and the folder is on the path."""


def test_lelib_imports():
    import lelib
    assert hasattr(lelib, "doubleMotor")


def test_mqttlib_imports():
    import mqttlib
    assert hasattr(mqttlib, "MQTTClient")
