"""Utilities for calculating gearbox ratios for the F4 car model.

The gear ratios are derived from the engine map stored in
``data/F4_T421_2022/engine_model/engine_map_414_F4_Gen2.json``.  Each total
ratio is the product of the individual gear ratio and the final drive ratio.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

ENGINE_MAP_PATH = (
    Path(__file__).resolve()
    .parent
    .parent
    / "data"
    / "F4_T421_2022"
    / "engine_model"
    / "engine_map_414_F4_Gen2.json"
)

HP_TO_WATTS = 745.699872


def load_engine_map(path: Path = ENGINE_MAP_PATH) -> Mapping[str, object]:
    """Load the engine map JSON file.

    Parameters
    ----------
    path:
        Optional override for the engine map path. Defaults to the repository
        data file bundled alongside the project.
    """

    with path.open() as json_file:
        return json.load(json_file)


def compute_total_gear_ratios(engine_map: Mapping[str, object]) -> Dict[int, float]:
    """Compute the overall gear ratio for each gear.

    The engine map defines a ``Final Ratio`` and per-gear ratios under the
    ``Gear Ratio`` key. The total gear ratio for gear *n* is calculated as::

        total_ratio[n] = gear_ratio[n] * final_ratio

    Parameters
    ----------
    engine_map:
        Parsed JSON contents from ``engine_map_414_F4_Gen2.json``.

    Returns
    -------
    Dict[int, float]
        Mapping of gear number to the overall ratio, preserving the order in
        which the gears appear in the JSON file.
    """

    try:
        final_ratio = float(engine_map["Final Ratio"])
        gear_ratios = engine_map["Gear Ratio"]
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(f"Missing required key '{missing_key}' in engine map") from exc

    total_ratios: Dict[int, float] = {}
    for gear, ratio in gear_ratios.items():
        total_ratios[int(gear)] = float(ratio) * final_ratio
    return total_ratios


def _build_power_interpolator(
    rpm_power_map: Iterable[Mapping[str, float]],
) -> Tuple[Sequence[float], Sequence[float], callable]:
    """Create a helper that interpolates power (HP) for any given RPM.

    Parameters
    ----------
    rpm_power_map:
        Iterable of mappings each containing ``"rpm"`` and ``"power_hp"`` keys.

    Returns
    -------
    Tuple[Sequence[float], Sequence[float], callable]
        Sorted RPM samples, corresponding power values, and a lookup function
        ``lookup(rpm: float) -> float`` that linearly interpolates power.
    """

    sorted_points = sorted(rpm_power_map, key=lambda point: point["rpm"])
    rpms: List[float] = [float(point["rpm"]) for point in sorted_points]
    powers: List[float] = [float(point["power_hp"]) for point in sorted_points]

    def lookup(rpm: float) -> float:
        if rpm <= rpms[0]:
            return powers[0]
        if rpm >= rpms[-1]:
            return powers[-1]

        insert_at = bisect_left(rpms, rpm)
        low_rpm, high_rpm = rpms[insert_at - 1], rpms[insert_at]
        low_power, high_power = powers[insert_at - 1], powers[insert_at]

        slope = (high_power - low_power) / (high_rpm - low_rpm)
        return low_power + slope * (rpm - low_rpm)

    return rpms, powers, lookup


def compute_power_by_speed(
    engine_map: Mapping[str, object],
    *,
    wheel_radius_m: float = 0.287,
    min_speed_kmh: int = 1,
    max_speed_kmh: int = 300,
    step_kmh: int = 1,
) -> MutableMapping[int, List[Tuple[int, float]]]:
    """Estimate engine power at incremental vehicle speeds for each gear.

    The function converts each speed step into wheel RPM (using the provided
    wheel radius) and then into engine RPM via the total gear ratio. Power is
    obtained by linearly interpolating the engine map's ``rpm_power_map``.
    Speeds that result in RPM outside the map's range are clamped to the
    nearest available value.

    Parameters
    ----------
    engine_map:
        Parsed JSON contents from ``engine_map_414_F4_Gen2.json``.
    wheel_radius_m:
        Tyre radius in meters. Defaults to 0.287 m (287 mm).
    min_speed_kmh, max_speed_kmh, step_kmh:
        Define the inclusive speed range (in km/h) for which to compute power
        estimates.

    Returns
    -------
    MutableMapping[int, List[Tuple[int, float]]]
        Mapping from gear number to a list of ``(speed_kmh, power_hp)`` pairs.
    """

    total_ratios = compute_total_gear_ratios(engine_map)
    try:
        rpm_power_map = engine_map["rpm_power_map"]
    except KeyError as exc:
        raise KeyError("Missing required key 'rpm_power_map' in engine map") from exc

    _, _, power_lookup = _build_power_interpolator(rpm_power_map)

    power_by_gear: MutableMapping[int, List[Tuple[int, float]]] = {}
    speeds = range(min_speed_kmh, max_speed_kmh + 1, step_kmh)
    circumference = 2 * math.pi * wheel_radius_m

    for gear, ratio in total_ratios.items():
        gear_power: List[Tuple[int, float]] = []
        for speed_kmh in speeds:
            speed_mps = speed_kmh / 3.6
            wheel_rpm = (speed_mps / circumference) * 60
            engine_rpm = wheel_rpm * ratio
            power_hp = power_lookup(engine_rpm)
            gear_power.append((speed_kmh, power_hp))

        power_by_gear[gear] = gear_power

    return power_by_gear


def compute_force_by_speed(
    engine_map: Mapping[str, object],
    *,
    wheel_radius_m: float = 0.287,
    min_speed_kmh: int = 1,
    max_speed_kmh: int = 300,
    step_kmh: int = 1,
) -> MutableMapping[int, List[Tuple[int, float]]]:
    """Compute tractive force (N) at incremental speeds for each gear.

    Power (in horsepower) is converted to watts and divided by vehicle speed
    (in m/s) to obtain force. Power is sourced from
    :func:`compute_power_by_speed`, which performs the RPM conversion and
    interpolation.

    Parameters
    ----------
    engine_map:
        Parsed JSON contents from ``engine_map_414_F4_Gen2.json``.
    wheel_radius_m:
        Tyre radius in meters. Defaults to 0.287 m (287 mm).
    min_speed_kmh, max_speed_kmh, step_kmh:
        Define the inclusive speed range (in km/h) for which to compute force
        estimates.

    Returns
    -------
    MutableMapping[int, List[Tuple[int, float]]]
        Mapping from gear number to a list of ``(speed_kmh, force_newtons)``
        pairs.
    """

    power_by_gear = compute_power_by_speed(
        engine_map,
        wheel_radius_m=wheel_radius_m,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        step_kmh=step_kmh,
    )

    force_by_gear: MutableMapping[int, List[Tuple[int, float]]] = {}
    for gear, samples in power_by_gear.items():
        gear_forces: List[Tuple[int, float]] = []
        for speed_kmh, power_hp in samples:
            speed_mps = speed_kmh / 3.6
            power_watts = power_hp * HP_TO_WATTS
            force_newtons = power_watts / speed_mps
            gear_forces.append((speed_kmh, force_newtons))

        force_by_gear[gear] = gear_forces

    return force_by_gear


def plot_force_by_speed(
    force_by_gear: Mapping[int, Sequence[Tuple[int, float]]],
    *,
    title: str | None = None,
):
    """Plot force vs. speed curves for each gear.

    Parameters
    ----------
    force_by_gear:
        Mapping of gear number to an iterable of ``(speed_kmh, force_newtons)``
        pairs.
    title:
        Optional plot title.

    Returns
    -------
    (Figure, Axes)
        Matplotlib figure and axes populated with the force curves.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting force curves"
        ) from exc

    fig, ax = plt.subplots()

    for gear in sorted(force_by_gear):
        speeds, forces = zip(*force_by_gear[gear])
        ax.plot(speeds, forces, label=f"Gear {gear}")

    ax.set_xlabel("Velocidad (km/h)")
    ax.set_ylabel("Fuerza (N)")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig, ax


if __name__ == "__main__":
    engine_map = load_engine_map()
    total_ratios = compute_total_gear_ratios(engine_map)
    for gear in sorted(total_ratios):
        print(f"Gear {gear}: {total_ratios[gear]:.3f}")

    force_curves = compute_force_by_speed(engine_map)
    fig, _ = plot_force_by_speed(force_curves, title="Fuerza vs Velocidad por marcha")
    fig.show()