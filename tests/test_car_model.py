import math
import sys
from pathlib import Path

import pytest

# Ensure project root is on the import path for local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.car_model import (
    HP_TO_WATTS,
    compute_force_by_speed,
    compute_power_by_speed,
    compute_total_gear_ratios,
    compute_shift_rpms,
    force_plot_html,
    load_engine_map,
    plot_force_by_speed,
    _speed_from_engine_rpm,
    RPM_LIMIT,
    select_best_gear_by_speed,
)


def test_compute_total_gear_ratios_matches_engine_map():
    engine_map = load_engine_map()
    total_ratios = compute_total_gear_ratios(engine_map)

    expected = {
        1: 7.75,
        2: 5.8125,
        3: 4.65,
        4: 3.875,
        5: 3.224,
        6: 2.697,
    }

    assert total_ratios.keys() == expected.keys()
    for gear, ratio in expected.items():
        assert total_ratios[gear] == pytest.approx(ratio, rel=1e-6)


def test_compute_power_by_speed_interpolates_and_clamps():
    engine_map = load_engine_map()
    power_by_gear = compute_power_by_speed(engine_map, min_speed_kmh=1, max_speed_kmh=120)

    # Six gears should be present with 120 speed samples each
    assert set(power_by_gear.keys()) == {1, 2, 3, 4, 5, 6}
    for gear_curve in power_by_gear.values():
        assert len(gear_curve) == 120
        assert gear_curve[0][0] == 1
        assert gear_curve[-1][0] == 120

    # Spot-check interpolation inside the rpm map (3rd gear at 50 km/h)
    speed_50_power = dict(power_by_gear[3])[50]
    assert speed_50_power == pytest.approx(32.6607, rel=1e-4)

    # Spot-check interpolation near the top of the rpm map (3rd gear at 120 km/h)
    speed_120_power = dict(power_by_gear[3])[120]
    assert speed_120_power == pytest.approx(139.359, rel=1e-3)

    # Beyond map RPM should clamp to the final value (1st gear at 120 km/h)
    speed_120_gear1_power = dict(power_by_gear[1])[120]
    assert speed_120_gear1_power == pytest.approx(136.0, rel=1e-6)


def test_compute_force_by_speed_converts_hp_and_scales_with_speed():
    engine_map = load_engine_map()
    force_by_gear = compute_force_by_speed(engine_map, min_speed_kmh=1, max_speed_kmh=120)

    assert set(force_by_gear.keys()) == {1, 2, 3, 4, 5, 6}
    for force_curve in force_by_gear.values():
        assert len(force_curve) == 120
        assert force_curve[0][0] == 1
        assert force_curve[-1][0] == 120

    # Check that force matches power conversion (third gear at 50 km/h)
    power_hp = dict(compute_power_by_speed(engine_map, min_speed_kmh=50, max_speed_kmh=50)[3])[50]
    expected_force = (power_hp * HP_TO_WATTS) / (50 / 3.6)
    computed_force = dict(force_by_gear[3])[50]
    assert computed_force == pytest.approx(expected_force, rel=1e-6)

    # Force computation should align with converted power at other speeds
    power_hp_100 = dict(compute_power_by_speed(engine_map, min_speed_kmh=100, max_speed_kmh=100)[3])[100]
    expected_force_100 = (power_hp_100 * HP_TO_WATTS) / (100 / 3.6)
    computed_force_100 = dict(force_by_gear[3])[100]
    assert computed_force_100 == pytest.approx(expected_force_100, rel=1e-6)


def test_plot_force_by_speed_creates_figure_with_lines():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    engine_map = load_engine_map()
    force_by_gear = compute_force_by_speed(engine_map, min_speed_kmh=10, max_speed_kmh=20)

    fig, ax = plot_force_by_speed(force_by_gear, title="Test Force Curve")

    assert fig is not None
    assert ax.get_xlabel() == "Velocidad (km/h)"
    assert ax.get_ylabel() == "Fuerza (N)"
    assert ax.get_title() == "Test Force Curve"
    # One line per gear
    assert len(ax.get_lines()) == 6


def test_force_plot_html_contains_embedded_image():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    engine_map = load_engine_map()
    force_by_gear = compute_force_by_speed(engine_map, min_speed_kmh=10, max_speed_kmh=20)

    html = force_plot_html(force_by_gear, title="HTML Force Curve")

    assert "<html>" in html
    assert "HTML Force Curve" in html
    assert "data:image/png;base64" in html
    assert "<img" in html


def test_select_best_gear_by_speed_picks_max_force():
    engine_map = load_engine_map()
    force_by_gear = compute_force_by_speed(engine_map, min_speed_kmh=10, max_speed_kmh=120, step_kmh=10)

    best_gears = select_best_gear_by_speed(
        engine_map, min_speed_kmh=10, max_speed_kmh=120, step_kmh=10
    )

    # Ensure every speed step has a best gear
    assert [speed for speed, _, _ in best_gears] == list(range(10, 121, 10))

    # Validate against a manual argmax of the force curves
    expected = []
    for speed_kmh in range(10, 121, 10):
        best_gear = max(
            force_by_gear.items(),
            key=lambda item: dict(item[1])[speed_kmh],
        )[0]
        best_force = dict(force_by_gear[best_gear])[speed_kmh]
        expected.append((speed_kmh, best_gear, best_force))

    assert best_gears == expected


def test_select_best_gear_stops_at_top_gear_limiter():
    engine_map = load_engine_map()
    ratios = compute_total_gear_ratios(engine_map)
    top_gear = max(ratios)
    limiter_speed = _speed_from_engine_rpm(RPM_LIMIT, ratios[top_gear], 0.287)
    expected_max_speed = min(300, int(math.floor(limiter_speed)))

    best_gears = select_best_gear_by_speed(
        engine_map, min_speed_kmh=1, max_speed_kmh=300, step_kmh=1
    )

    assert best_gears[0][0] == 1
    assert best_gears[-1][0] == expected_max_speed
    assert [speed for speed, _, _ in best_gears] == list(
        range(1, expected_max_speed + 1)
    )
    assert best_gears[-1][1] == top_gear


def test_compute_shift_rpms_identifies_crossover_points():
    engine_map = load_engine_map()

    shift_points = compute_shift_rpms(
        engine_map, min_speed_kmh=1, max_speed_kmh=220, step_kmh=1
    )

    expected = [
        (1, 2, 86, 6100),
        (2, 3, 114, 6100),
        (3, 4, 141, 6059.81),
        (4, 5, 169, 6052.65),
        (5, 6, 203, 6048.92),
    ]

    # Validate shift sequence and crossover speeds
    assert [(f, t, s) for f, t, s, _ in shift_points] == [
        (f, t, s) for f, t, s, _ in expected
    ]

    # Check RPM values with a generous tolerance to allow for interpolation differences
    for (_, _, _, rpm), (_, _, _, expected_rpm) in zip(shift_points, expected):
        assert rpm == pytest.approx(expected_rpm, rel=1e-4)