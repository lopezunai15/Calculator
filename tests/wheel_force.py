"""Wheel force calculation utilities.

This module calculates the wheel force across every gear at 1 km/h steps.
It reads gear ratios and power curve data from the bundled
``engine_map_414_F4_Gen2.json`` file so the calculation uses the real
vehicle data instead of in-code placeholders. Power values in the input
must be expressed in horsepower and will internally be converted to watts
for force calculations.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

HP_TO_WATTS = 745.7


@dataclass
class GearSample:
    """Computed wheel-force data for a single speed point."""

    gear: int
    speed_kmh: float
    engine_rpm: float
    power_kw: float
    wheel_force_n: float


@dataclass
class VehicleData:
    """Container for the inputs required to compute wheel force."""

    final_ratio: float
    gear_ratios: Sequence[float]
    tire_radius_m: float
    # Power curve as a list of [rpm, horsepower] points.
    power_curve_hp: Sequence[Sequence[float]]

    @classmethod
    def from_json(cls, path: Path) -> "VehicleData":
        raw = json.loads(path.read_text())
        try:
            tire_radius_m = raw["tire_radius_mm"] / 1000
        except KeyError as exc:
            raise KeyError("JSON must contain 'tire_radius_mm'") from exc
        return cls(
            final_ratio=raw["final_ratio"],
            gear_ratios=raw["gear_ratios"],
            tire_radius_m=tire_radius_m,
            power_curve_hp=raw["power_curve_hp"],
        )


def hp_to_kw(horsepower: float) -> float:
    return horsepower * HP_TO_WATTS / 1000


def speed_kmh_to_mps(speed_kmh: float) -> float:
    return speed_kmh / 3.6


def engine_rpm(speed_kmh: float, total_gear_ratio: float, tire_radius_m: float) -> float:
    wheel_rad_s = speed_kmh_to_mps(speed_kmh) / tire_radius_m
    engine_rad_s = wheel_rad_s * total_gear_ratio
    return engine_rad_s * 60 / (2 * math.pi)


def interpolate_power_kw(power_curve_hp: Sequence[Sequence[float]], rpm: float) -> float:
    """Linearly interpolate the power curve expressed in horsepower.

    If the RPM is outside the provided range, 0 kW is returned to indicate
    that the engine is not operating in a defined region.
    """

    points = sorted((float(r), float(hp)) for r, hp in power_curve_hp)
    rpms = [r for r, _ in points]
    if rpm < rpms[0] or rpm > rpms[-1]:
        return 0.0

    for (rpm_low, hp_low), (rpm_high, hp_high) in zip(points, points[1:]):
        if rpm_low <= rpm <= rpm_high:
            if rpm_high == rpm_low:
                return hp_to_kw(hp_low)
            fraction = (rpm - rpm_low) / (rpm_high - rpm_low)
            horsepower = hp_low + fraction * (hp_high - hp_low)
            return hp_to_kw(horsepower)

    return 0.0


def compute_gear_samples(
    vehicle: VehicleData, gear_index: int, speed_step_kmh: int = 1
) -> List[GearSample]:
    total_ratio = vehicle.gear_ratios[gear_index] * vehicle.final_ratio
    max_rpm = max(r for r, _ in vehicle.power_curve_hp)
    max_speed = rpm_to_speed_kmh(max_rpm, total_ratio, vehicle.tire_radius_m)

    samples: List[GearSample] = []
    speed = speed_step_kmh
    while speed <= max_speed:
        rpm = engine_rpm(speed, total_ratio, vehicle.tire_radius_m)
        power_kw = interpolate_power_kw(vehicle.power_curve_hp, rpm)
        speed_mps = speed_kmh_to_mps(speed)
        wheel_force = 0.0 if speed_mps == 0 else (power_kw * 1000) / speed_mps
        samples.append(
            GearSample(
                gear=gear_index + 1,
                speed_kmh=speed,
                engine_rpm=rpm,
                power_kw=power_kw,
                wheel_force_n=wheel_force,
            )
        )
        speed += speed_step_kmh
    return samples


def rpm_to_speed_kmh(rpm: float, total_gear_ratio: float, tire_radius_m: float) -> float:
    wheel_rad_s = rpm * (2 * math.pi) / 60 / total_gear_ratio
    speed_mps = wheel_rad_s * tire_radius_m
    return speed_mps * 3.6


def compute_wheel_force(vehicle: VehicleData, speed_step_kmh: int = 1) -> Dict[int, List[GearSample]]:
    return {
        gear_index + 1: compute_gear_samples(vehicle, gear_index, speed_step_kmh)
        for gear_index in range(len(vehicle.gear_ratios))
    }


def suggest_shift_points(samples: Dict[int, List[GearSample]]) -> Dict[int, float]:
    """Suggest gear change points based on wheel force comparison.

    The shift point between gear N and N+1 is the first speed where the
    wheel force of gear N+1 is greater than or equal to gear N.
    """

    shift_points: Dict[int, float] = {}
    for gear in range(1, len(samples)):
        current = {round(s.speed_kmh, 3): s for s in samples[gear]}
        nxt = {round(s.speed_kmh, 3): s for s in samples[gear + 1]}
        overlap_speeds = sorted(set(current.keys()) & set(nxt.keys()))
        for speed in overlap_speeds:
            if nxt[speed].wheel_force_n >= current[speed].wheel_force_n:
                shift_points[gear] = speed
                break
    return shift_points


def _example_vehicle() -> VehicleData:
    engine_map_path = (
        Path(__file__).resolve()
        .parent
        .parent
        / "data"
        / "F4_T421_2022"
        / "engine_model"
        / "engine_map_414_F4_Gen2.json"
    )
    return VehicleData.from_json(engine_map_path)


def main() -> None:
    vehicle = _example_vehicle()
    samples = compute_wheel_force(vehicle)
    shift_points = suggest_shift_points(samples)

    for gear, gear_samples in samples.items():
        print(f"\nGear {gear} (total ratio {vehicle.gear_ratios[gear-1]*vehicle.final_ratio:.2f})")
        for sample in gear_samples[:5]:  # print first few samples for brevity
            print(
                f"  {sample.speed_kmh:>3.0f} km/h -> {sample.engine_rpm:>6.0f} rpm, "
                f"{sample.power_kw:5.1f} kW, {sample.wheel_force_n:7.1f} N"
            )

    print("\nSuggested shift speeds (km/h):")
    for gear, speed in shift_points.items():
        print(f"  Gear {gear} -> {gear+1}: {speed:.0f} km/h")


if __name__ == "__main__":
    main()