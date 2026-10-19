"""The sumo behavior tree (py_trees).

ROOT (Selector)
├─ DEFEATED (Sequence, latched until control/reset)
│  ├─ [Cond] defeat key (d) pressed
│  ├─ [Act]  stop motors, park flipper
│  └─ [Act]  publish cmd/sound "sad" once
└─ ACTIVE (Parallel: DRIVE and FLIPPER both tick every cycle)
   ├─ DRIVE (Selector)
   │  ├─ AVOID_EDGE:   blue seen (sensor at the back) -> forward-only levers, else stop
   │  ├─ TELEOP_DRIVE: levers fresh and moved -> edge sensor ok -> tank drive
   │  ├─ HOLD_GROUND:  edge sensor ok -> opponent < hold_ground_cm -> push forward
   │  └─ IDLE:         stop
   └─ FLIPPER (Selector)
      ├─ MANUAL_FIRE: fire key fresh -> ready -> fire
      ├─ MANUAL_MODE: mode == manual -> do nothing (blocks auto-fire)
      ├─ AUTO_FIRE:   opponent < auto_fire_cm -> ready -> not avoiding edge -> fire
      └─ REST:        home the flipper once it has finished its stroke

Behaviours don't publish anything themselves. They write the outputs for this
tick into a Context, and brain.Brain publishes them after the tick.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import py_trees
from py_trees.common import Status

from brain.world import World

log = logging.getLogger("brain")


class Context:
    """Shared state the behaviours read and write. One per brain."""

    def __init__(self, cfg: dict[str, Any], world: World) -> None:
        self.cfg = cfg
        self.world = world
        self.now = 0.0
        # outputs for the current tick
        self.motors: tuple[float, float] = (0.0, 0.0)
        self.src_t: float | None = None      # timestamp of the input behind self.motors
        self.flipper_cmds: list[str] = []
        self.sounds: list[str] = []
        # latched / remembered across ticks
        self.defeated = False
        self.song_sent = False
        self.parked = False
        self.last_blue: float | None = None
        self.avoiding = False
        self.last_fire: float | None = None
        self.home_pending = True             # home the flipper once at startup

    def begin_tick(self, now: float) -> None:
        self.now = now
        self.motors, self.src_t = (0.0, 0.0), None
        self.flipper_cmds, self.sounds = [], []

    def reset(self) -> None:
        """control/reset: clear the defeated latch."""
        self.defeated = self.song_sent = self.parked = False
        self.last_blue, self.avoiding = None, False
        self.home_pending = True
        self.world.reset()
        self.world.consume_fire()


class Check(py_trees.behaviour.Behaviour):
    """Condition leaf: SUCCESS if fn(ctx) is true, else FAILURE."""

    def __init__(self, name: str, ctx: Context, fn: Callable[[Context], bool]) -> None:
        super().__init__(name)
        self.ctx, self.fn = ctx, fn

    def update(self) -> Status:
        return Status.SUCCESS if self.fn(self.ctx) else Status.FAILURE


class Do(py_trees.behaviour.Behaviour):
    """Action leaf: runs fn(ctx), which returns a Status."""

    def __init__(self, name: str, ctx: Context, fn: Callable[[Context], Status]) -> None:
        super().__init__(name)
        self.ctx, self.fn = ctx, fn

    def update(self) -> Status:
        return self.fn(self.ctx)


# ---- DEFEATED ---------------------------------------------------------------

def check_defeated(ctx: Context) -> bool:
    if not ctx.defeated and ctx.world.manual_defeat:
        log.warning("DEFEATED (defeat key)")
        ctx.defeated = True
    return ctx.defeated


def stop_and_park(ctx: Context) -> Status:
    ctx.motors = (0.0, 0.0)
    if not ctx.parked:
        ctx.flipper_cmds.append("home")
        ctx.parked = True
    return Status.SUCCESS


def sad_song_once(ctx: Context) -> Status:
    if not ctx.song_sent:
        ctx.sounds.append("sad")
        ctx.song_sent = True
    return Status.SUCCESS


# ---- DRIVE ------------------------------------------------------------------

def edge_detected(ctx: Context) -> bool:
    """Blue seen now, or within edge.clear_s ago."""
    if ctx.world.on_blue(ctx.now):
        ctx.last_blue = ctx.now
    if ctx.last_blue is not None and ctx.now - ctx.last_blue >= ctx.cfg["edge"]["clear_s"]:
        ctx.last_blue = None
    ctx.avoiding = ctx.last_blue is not None
    return ctx.avoiding


def levers_moved(ctx: Context) -> bool:
    joy = ctx.world.joystick(ctx.now)
    return joy is not None and max(abs(joy[0]), abs(joy[1])) > ctx.cfg["joystick"]["deadzone"]


def lever_speeds(ctx: Context) -> tuple[float, float]:
    """Wheel speeds from the levers, scaled by joystick.max_speed."""
    left, right = ctx.world.joy
    s = ctx.cfg["joystick"]["max_speed"]
    return left * s, right * s


def forward_only(ctx: Context) -> Status:
    """The color sensor is at the back, so the tape is behind us: stop, and let
    the levers move the robot forward only (no wheel may turn backwards)."""
    left, right = lever_speeds(ctx) if levers_moved(ctx) else (0.0, 0.0)
    ctx.motors = (max(0.0, left), max(0.0, right))
    ctx.src_t = ctx.world.color_t
    return Status.RUNNING


def edge_sensor_ok(ctx: Context) -> bool:
    return not ctx.cfg["brain"]["require_color_for_drive"] or ctx.world.color_fresh(ctx.now)


def tank_drive(ctx: Context) -> Status:
    ctx.motors = lever_speeds(ctx)
    ctx.src_t = ctx.world.joy_t
    return Status.RUNNING


def opponent_close(ctx: Context) -> bool:
    d = ctx.world.distance(ctx.now)
    return d is not None and d < ctx.cfg["ultrasonic"]["hold_ground_cm"]


def push(ctx: Context) -> Status:
    s = ctx.cfg["hold_ground"]["push_speed"]
    ctx.motors = (s, s)
    ctx.src_t = ctx.world.us_t
    return Status.RUNNING


def idle(ctx: Context) -> Status:
    ctx.motors = (0.0, 0.0)
    return Status.SUCCESS


# ---- FLIPPER ----------------------------------------------------------------

def flipper_ready(ctx: Context) -> bool:
    return ctx.last_fire is None or ctx.now - ctx.last_fire >= ctx.cfg["flipper"]["cooldown_s"]


def fire(ctx: Context) -> Status:
    ctx.flipper_cmds.append("fire")
    ctx.last_fire = ctx.now
    ctx.home_pending = True
    ctx.world.start_blanking(ctx.now)
    ctx.world.consume_fire()
    log.info("FIRE")
    return Status.SUCCESS


def opponent_in_range(ctx: Context) -> bool:
    d = ctx.world.distance(ctx.now)
    return d is not None and d < ctx.cfg["ultrasonic"]["auto_fire_cm"]


def rest(ctx: Context) -> Status:
    stroke_done = ctx.last_fire is None or ctx.now - ctx.last_fire >= ctx.cfg["flipper"]["sweep_s"]
    if ctx.home_pending and stroke_done:
        ctx.flipper_cmds.append("home")
        ctx.home_pending = False
    return Status.SUCCESS


# ---- assembly ---------------------------------------------------------------

def build_tree(ctx: Context) -> py_trees.composites.Selector:
    """Build the tree described in the module docstring."""
    S, Sel = py_trees.composites.Sequence, py_trees.composites.Selector

    defeated = S("DEFEATED", memory=False, children=[
        Check("defeat_key", ctx, check_defeated),
        Do("stop_and_park", ctx, stop_and_park),
        Do("sad_song_once", ctx, sad_song_once),
    ])
    drive = Sel("DRIVE", memory=False, children=[
        S("AVOID_EDGE", memory=False, children=[
            Check("edge_detected", ctx, edge_detected),
            Do("forward_only", ctx, forward_only),
        ]),
        S("TELEOP_DRIVE", memory=False, children=[
            Check("levers_fresh", ctx, levers_moved),
            Check("edge_sensor_ok", ctx, edge_sensor_ok),
            Do("tank_drive", ctx, tank_drive),
        ]),
        S("HOLD_GROUND", memory=False, children=[
            Check("edge_sensor_ok", ctx, edge_sensor_ok),
            Check("opponent_close", ctx, opponent_close),
            Do("push", ctx, push),
        ]),
        Do("IDLE", ctx, idle),
    ])
    flipper = Sel("FLIPPER", memory=False, children=[
        S("MANUAL_FIRE", memory=False, children=[
            Check("fire_key_fresh", ctx, lambda c: c.world.fire_requested(c.now)),
            Check("flipper_ready", ctx, flipper_ready),
            Do("fire", ctx, fire),
        ]),
        S("MANUAL_MODE", memory=False, children=[
            Check("mode_is_manual", ctx, lambda c: c.world.flipper_mode == "manual"),
            Do("hold_fire", ctx, lambda c: Status.SUCCESS),
        ]),
        S("AUTO_FIRE", memory=False, children=[
            Check("opponent_in_range", ctx, opponent_in_range),
            Check("flipper_ready", ctx, flipper_ready),
            Check("not_avoiding_edge", ctx, lambda c: not c.avoiding),
            Do("fire", ctx, fire),
        ]),
        Do("REST", ctx, rest),
    ])
    active = py_trees.composites.Parallel(
        "ACTIVE",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
        children=[drive, flipper],
    )
    return Sel("ROOT", memory=False, children=[defeated, active])


def active_nodes(root: py_trees.composites.Selector) -> tuple[str, str]:
    """(drive node, flipper node) chosen on the last tick, e.g. ("AVOID_EDGE", "REST")."""
    top = root.current_child
    if top is None or top.name == "DEFEATED":
        return "DEFEATED", "DEFEATED"
    drive, flipper = top.children
    return (drive.current_child.name if drive.current_child else "?",
            flipper.current_child.name if flipper.current_child else "?")
