"""The base scenario of this project: a person walking on their own in the Indego (indego_sim, gait_mode "walk").

Sagittal-plane model, no crutches or help, 30 ms long-loop reaction delay; the person's walking parameters come from
indego_aan.humans ("robust": copes with its own deficits, default; "tuned": indego_sim's fresh-walking set). The exo
is driven by whatever controller the caller steps it with.
"""
from __future__ import annotations

from typing import Optional

from indego_sim.config import SimConfig, SupportParams, VolitionParams
from indego_sim.human.sci import get_preset
from indego_sim.model.builder import build_model
from indego_sim.sim import HumanExoSim

from .humans import walk_params

DEFAULT_SUBJECT = "sci_t12_aisD"
REACTION_DELAY = 0.03          # s long-loop delay the tuned walking assumes (indego_sim docs/STATUS.md)


def walking_subject(name: str = DEFAULT_SUBJECT, delay: float = REACTION_DELAY):
    subject = get_preset(name)
    subject.long_loop_delay = delay
    return subject


def build_walking_model(name: str = DEFAULT_SUBJECT, delay: float = REACTION_DELAY):
    """(m, info) for the scenario; build once and pass to make_walking_sim(model=...) to save the build time.
    Never pass a running sim's own model (sim.m) instead: the person's controller scales the muscles in place (SCI
    atrophy, passive stiffness) on the model it is given, so a reused sim.m is weakened again in every new sim."""
    return build_model(subject=walking_subject(name, delay), sim=SimConfig(planar=True))


def make_walking_sim(seed: int = 0, name: str = DEFAULT_SUBJECT, delay: float = REACTION_DELAY,
                     walk_at: float = 0.5, settle: float = 1.5, model: Optional[tuple] = None,
                     log_rate_hz: float = 20.0, human: str = "robust"):
    """A HumanExoSim of the person walking on their own; returns (sim, first observation). The person starts walking
    `walk_at` s after the settle."""
    vol = VolitionParams(gait_mode="walk", walk=walk_params(name, human))
    sim = HumanExoSim(subject=walking_subject(name, delay), support=SupportParams(kind="none"), volition=vol,
                      sim=SimConfig(planar=True), schedule=[(walk_at, "walk")], seed=seed, model=model,
                      log_muscles=False, log_rate_hz=log_rate_hz)
    obs = sim.reset(settle=settle)
    return sim, obs
