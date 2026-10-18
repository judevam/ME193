"""Run the brain: python -m brain.main"""
from brain.brain import Brain
from common.bus import MqttBus
from common.config import load_config, setup_logging
from common.runtime import Rate, StopFlag


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, "brain")
    stop = StopFlag()
    bus = MqttBus(cfg, "brain")
    brain = Brain(cfg, bus)
    rate = Rate(cfg["brain"]["tick_hz"], stop)
    log.info("brain running at %s Hz", cfg["brain"]["tick_hz"])
    try:
        while not stop.is_set():
            brain.step()
            if rate.sleep():
                break
    finally:
        brain.stop_motors()
        bus.close()  # bridges also stop on our "offline" Last Will
        log.info("brain stopped")


if __name__ == "__main__":
    main()
