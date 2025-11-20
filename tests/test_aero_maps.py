"""CLI para calcular valores aerodinámicos AB, CD y CL a partir de los aero mapas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


AERO_DIR = Path(__file__).resolve().parents[1] / "data" / "F4_T421_2022" / "aero_model"
MAP_SPECS = (
    ("AB", "AB_aero_map_v1.json"),
    ("CD", "CD_aero_map_v1.json"),
    ("CL", "CL_aero_map_v1.json"),
)


def evaluate_polynomial(coefficients: Iterable[float], powers: Iterable[Dict[str, int]], variables: Dict[str, float]) -> float:
    """Evalúa un polinomio definido por coeficientes y potencias.

    Los diccionarios de potencias usan como clave el nombre de la variable
    (por ejemplo, "fw", "rw", "frh", "rrh").
    """

    total = 0.0
    for coefficient, power_set in zip(coefficients, powers):
        term = coefficient
        for variable, exponent in power_set.items():
            term *= variables[variable] ** exponent
        total += term
    return total


def validate_ranges(model: Dict, provided_values: Dict[str, float]) -> None:
    """Comprueba que los valores del usuario respetan los rangos definidos."""

    for variable, settings in model["variables"].items():
        lower, upper = settings["range"]
        value = provided_values[variable]
        if not lower <= value <= upper:
            raise ValueError(
                f"El valor de {variable}={value} está fuera del rango permitido [{lower}, {upper}]"
            )


def evaluate_map(
    map_data: Dict,
    prefix: str,
    wing_inputs: Dict[str, float],
    height_inputs: Dict[str, float],
    gurney_installed: bool,
) -> float:
    """Calcula el valor objetivo (AB, CD o CL) para un mapa concreto."""

    model_key = "with_gurney" if gurney_installed else "without_gurney"
    models = map_data["models"][model_key]

    wing_model = models[f"{prefix}_wing_model"]
    height_model = models[f"{prefix}_height_model"]

    validate_ranges(wing_model, wing_inputs)
    validate_ranges(height_model, height_inputs)

    wing_result = evaluate_polynomial(
        wing_model["coefficients"], wing_model["powers"], wing_inputs
    )
    height_result = evaluate_polynomial(
        height_model["coefficients"], height_model["powers"], height_inputs
    )

    average_delta = (wing_result + height_result) / 2
    standard_value = map_data[f"{prefix}_standard"]

    if prefix == "AB":
        return standard_value * (1 + average_delta)

    return standard_value * average_delta


def compute_aero_values(fw: float, rw: float, frh: float, rrh: float, gurney: bool) -> Dict[str, float]:
    """Carga los mapas aerodinámicos y devuelve AB, CD y CL."""

    wing_inputs = {"fw": fw, "rw": rw}
    height_inputs = {"frh": frh, "rrh": rrh}

    results: Dict[str, float] = {}
    for prefix, filename in MAP_SPECS:
        with (AERO_DIR / filename).open("r", encoding="utf-8") as aero_file:
            map_data = json.load(aero_file)
        results[prefix] = evaluate_map(map_data, prefix, wing_inputs, height_inputs, gurney)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula AB, CD y CL a partir de FW, RW, FRH, RRH y presencia de gurney.",
    )
    parser.add_argument("--fw", type=float, required=True, help="Ángulo del front wing (deg)")
    parser.add_argument("--rw", type=float, required=True, help="Ángulo del rear wing (deg)")
    parser.add_argument("--frh", type=float, required=True, help="Ride height delantero (mm)")
    parser.add_argument("--rrh", type=float, required=True, help="Ride height trasero (mm)")
    parser.add_argument(
        "--gurney",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indica si el gurney está instalado",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = compute_aero_values(args.fw, args.rw, args.frh, args.rrh, args.gurney)
    print("Resultados aerodinámicos:")
    print(f"  AB: {values['AB']:.6f}")
    print(f"  CD: {values['CD']:.6f}")
    print(f"  CL: {values['CL']:.6f}")


if __name__ == "__main__":
    main()

"""Console input: python tests/test_aero_maps.py --fw 5 --rw 10 --frh 40 --rrh 15 --gurney """