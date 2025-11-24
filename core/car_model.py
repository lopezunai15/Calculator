"""Utilities for calculating gearbox ratios for the F4 car model.

The gear ratios are derived from the engine map stored in
``data/F4_T421_2022/engine_model/engine_map_414_F4_Gen2.json``.  Each total
ratio is the product of the individual gear ratio and the final drive ratio.
"""
from __future__ import annotations

import base64
import io
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
RPM_LIMIT = 6100


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
            engine_rpm = min(wheel_rpm * ratio, RPM_LIMIT)
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


def _engine_rpm_from_speed(
    speed_kmh: float, total_ratio: float, wheel_radius_m: float, rpm_limit: float
) -> float:
    """Convert vehicle speed (km/h) to engine RPM using the total gear ratio and clamp to the limiter."""

    speed_mps = speed_kmh / 3.6
    wheel_rpm = (speed_mps / (2 * math.pi * wheel_radius_m)) * 60
    return min(wheel_rpm * total_ratio, rpm_limit)


def _speed_from_engine_rpm(
    engine_rpm: float, total_ratio: float, wheel_radius_m: float
) -> float:
    """Convert engine RPM to vehicle speed (km/h) using the total gear ratio."""

    wheel_rpm = engine_rpm / total_ratio
    speed_mps = (wheel_rpm * (2 * math.pi * wheel_radius_m)) / 60
    return speed_mps * 3.6


def select_best_gear_by_speed(
    engine_map: Mapping[str, object],
    *,
    wheel_radius_m: float = 0.287,
    min_speed_kmh: int = 1,
    max_speed_kmh: int = 300,
    step_kmh: int = 1,
) -> List[Tuple[int, int, float]]:
    """Pick the gear and force that deliver the most tractive effort per speed step.

    Parameters
    ----------
    engine_map:
        Parsed JSON contents from ``engine_map_414_F4_Gen2.json``.
    wheel_radius_m:
        Tyre radius in meters. Defaults to 0.287 m (287 mm).
    min_speed_kmh, max_speed_kmh, step_kmh:
        Define the inclusive speed range (in km/h) for which to compute the
        best-gear selection.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of ``(speed_kmh, best_gear, best_force_newtons)`` tuples, where the
        gear is chosen by maximizing tractive force. Ties are broken by
        selecting the lowest gear number. The scan halts at the speed where top
        gear reaches the engine limiter to avoid producing unrealistic
        higher-speed samples.
    """

    ratios = compute_total_gear_ratios(engine_map)
    top_gear = max(ratios)
    top_gear_limiter_speed = _speed_from_engine_rpm(
        RPM_LIMIT, ratios[top_gear], wheel_radius_m
    )
    max_scan_speed = min(max_speed_kmh, int(math.floor(top_gear_limiter_speed)))

    if max_scan_speed < min_speed_kmh:
        raise ValueError(
            "Limiter speed for top gear is below the minimum speed to scan"
        )

    force_by_gear = compute_force_by_speed(
        engine_map,
        wheel_radius_m=wheel_radius_m,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        step_kmh=step_kmh,
    )

    best_gears: List[Tuple[int, int, float]] = []
    speeds = range(min_speed_kmh, max_scan_speed + 1, step_kmh)
    force_lookup = {gear: dict(samples) for gear, samples in force_by_gear.items()}

    for speed_kmh in speeds:
        best_gear: int | None = None
        best_force = -math.inf
        for gear, speed_to_force in force_lookup.items():
            force = speed_to_force[speed_kmh]
            if force > best_force or (
                math.isclose(force, best_force)
                and best_gear is not None
                and gear < best_gear
            ):
                best_force = force
                best_gear = gear

        if best_gear is None:
            raise RuntimeError(
                f"Failed to determine best gear for speed {speed_kmh} km/h"
            )

        best_gears.append((speed_kmh, best_gear, best_force))

    return best_gears


def compute_shift_rpms(
    engine_map: Mapping[str, object],
    *,
    wheel_radius_m: float = 0.287,
    min_speed_kmh: int = 1,
    max_speed_kmh: int = 300,
    step_kmh: int = 1,
) -> List[Tuple[int, int, int, float]]:
    """Determine the engine RPM where each upshift should occur.

    For each adjacent gear pair, the function scans the speed range to find the
    first speed where the higher gear produces equal or greater tractive force
    than the lower gear. That crossover speed represents the optimal upshift
    point when prioritizing maximum force delivery. If the lower gear reaches
    the rev limiter before the crossover, the shift is commanded at the limiter
    speed instead of the force-based point. The engine RPM at the selected
    speed is calculated using the total ratio of the lower gear and is clamped
    to the rev limit.

    Parameters
    ----------
    engine_map:
        Parsed JSON contents from ``engine_map_414_F4_Gen2.json``.
    wheel_radius_m:
        Tyre radius in meters. Defaults to 0.287 m (287 mm).
    min_speed_kmh, max_speed_kmh, step_kmh:
        Define the inclusive speed range (in km/h) over which to search for
        gear crossover points.

    Returns
    -------
    List[Tuple[int, int, int, float]]
        A list of ``(from_gear, to_gear, shift_speed_kmh, shift_rpm)`` tuples
        for each upshift. A ``RuntimeError`` is raised if no crossover is found
        for a gear pair within the scanned range.
    """

    force_by_gear = compute_force_by_speed(
        engine_map,
        wheel_radius_m=wheel_radius_m,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        step_kmh=step_kmh,
    )
    ratios = compute_total_gear_ratios(engine_map)

    shift_points: List[Tuple[int, int, int, float]] = []
    speeds = range(min_speed_kmh, max_speed_kmh + 1, step_kmh)

    for gear in sorted(force_by_gear):
        next_gear = gear + 1
        if next_gear not in force_by_gear:
            continue

        current_forces = dict(force_by_gear[gear])
        next_forces = dict(force_by_gear[next_gear])

        shift_speed: int | None = None
        for speed_kmh in speeds:
            if next_forces[speed_kmh] >= current_forces[speed_kmh]:
                shift_speed = speed_kmh
                break

        if shift_speed is None:
            raise RuntimeError(
                f"No crossover point found between gear {gear} and {next_gear}"
                f" within {min_speed_kmh}-{max_speed_kmh} km/h"
            )

        limiter_speed_kmh = _speed_from_engine_rpm(
            RPM_LIMIT, ratios[gear], wheel_radius_m
        )
        limiter_speed_step = int(math.ceil(limiter_speed_kmh))

        if limiter_speed_step < min_speed_kmh:
            limiter_speed_step = min_speed_kmh
        if limiter_speed_step > max_speed_kmh:
            limiter_speed_step = max_speed_kmh

        if limiter_speed_step < shift_speed:
            shift_speed = limiter_speed_step
            shift_rpm = float(RPM_LIMIT)
        else:
            shift_rpm = _engine_rpm_from_speed(
                shift_speed, ratios[gear], wheel_radius_m, RPM_LIMIT
            )
        shift_points.append((gear, next_gear, shift_speed, shift_rpm))

    return shift_points


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


def force_plot_html(
    force_by_gear: Mapping[int, Sequence[Tuple[int, float]]],
    *,
    title: str | None = None,
) -> str:
    """Render the force plot and return an embeddable HTML string.

    The plot is generated with :func:`plot_force_by_speed` and saved to an
    in-memory PNG, which is embedded as a data URI within minimal HTML. This
    allows consumers to persist or display the plot without managing image
    files separately.

    Parameters
    ----------
    force_by_gear:
        Mapping of gear number to an iterable of ``(speed_kmh, force_newtons)``
        pairs.
    title:
        Optional plot title.

    Returns
    -------
    str
        HTML string containing a base64-encoded ``<img>`` tag.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for rendering HTML force plots"
        ) from exc

    fig, _ = plot_force_by_speed(force_by_gear, title=title)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    img_tag = (
        f"<img alt='Force vs Speed plot' "
        f"src='data:image/png;base64,{encoded}' />"
    )

    body_content = img_tag if title is None else f"<h1>{title}</h1>\n{img_tag}"
    return f"<html><body>{body_content}</body></html>"


if __name__ == "__main__":
    engine_map = load_engine_map()
    total_ratios = compute_total_gear_ratios(engine_map)
    for gear in sorted(total_ratios):
        print(f"Gear {gear}: {total_ratios[gear]:.3f}")

    force_curves = compute_force_by_speed(engine_map)
    fig, _ = plot_force_by_speed(force_curves, title="Fuerza vs Velocidad por marcha")
    fig.show()

    best_gears = select_best_gear_by_speed(engine_map)
    print("Marcha óptima por velocidad (km/h -> marcha, fuerza):")
    for speed, gear, force in best_gears:
        print(f"{speed:3d} -> {gear} ({force:.1f} N)")

    shift_points = compute_shift_rpms(engine_map)
    print("\nPuntos de cambio de marcha (velocidad, rpm):")
    for from_gear, to_gear, speed, rpm in shift_points:
        print(f"{from_gear} -> {to_gear}: {speed} km/h @ {rpm:.0f} rpm")

    html_output = force_plot_html(
        force_curves, title="Fuerza vs Velocidad por marcha"
    )
    output_path = Path(__file__).parent / "force_vs_speed.html"
    output_path.write_text(html_output, encoding="utf-8")
    print(f"Force plot HTML saved to {output_path}")