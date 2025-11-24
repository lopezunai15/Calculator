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
    load_engine_map,
    plot_force_by_speed,
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