"""Genera contour plots interactivos para los modelos aerodinámicos.

Este script recorre los ficheros JSON dentro de ``data/F4_T421_2022/aero_model`` y
crea cuatro contour plots por cada mapa (wing/height con y sin gurney) respetando
los rangos de entrada definidos en cada modelo. El valor mostrado corresponde al
resultado completo calculado por ``test_aero_maps.py`` (combinando wing y height
según las fórmulas de AB/CD/CL) y no únicamente al polinomio individual.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from string import Template
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from test_aero_maps import AERO_DIR, MAP_SPECS, evaluate_map


DEFAULT_POINTS = 75
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def _midpoint_inputs(model: Dict) -> Dict[str, float]:
    """Devuelve los valores medios del rango de cada variable."""

    return {
        name: (settings["range"][0] + settings["range"][1]) / 2
        for name, settings in model["variables"].items()
    }


def _integer_axis(lower: float, upper: float, max_points: int) -> np.ndarray:
    """Devuelve valores enteros dentro del rango [lower, upper].

    Si el rango no contiene enteros o hay demasiados valores, se muestrean de forma
    uniforme hasta ``max_points`` elementos.
    """

    start = math.ceil(lower)
    end = math.floor(upper)
    if start > end:
        return np.linspace(lower, upper, max_points)

    values = np.arange(start, end + 1, dtype=float)
    if len(values) <= max_points:
        return values

    idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    return np.unique(values[idx])


def _grid_for_model(
    prefix: str, map_data: Dict, model_kind: str, with_gurney: bool, num_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye rejillas y evalúa el valor final del mapa en cada punto."""

    models = map_data["models"]["with_gurney" if with_gurney else "without_gurney"]
    wing_model = models[f"{prefix}_wing_model"]
    height_model = models[f"{prefix}_height_model"]

    varying_model = wing_model if model_kind == "wing" else height_model
    fixed_inputs = _midpoint_inputs(height_model if model_kind == "wing" else wing_model)

    variable_names = list(varying_model["variables"].keys())
    x_name, y_name = variable_names

    (x_lower, x_upper) = varying_model["variables"][x_name]["range"]
    (y_lower, y_upper) = varying_model["variables"][y_name]["range"]

    x_values = _integer_axis(x_lower, x_upper, num_points)
    y_values = _integer_axis(y_lower, y_upper, num_points)

    Z = np.empty((len(y_values), len(x_values)), dtype=float)
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            if model_kind == "wing":
                wing_inputs = {x_name: float(x), y_name: float(y)}
                height_inputs = fixed_inputs
            else:
                wing_inputs = fixed_inputs
                height_inputs = {x_name: float(x), y_name: float(y)}

            Z[i, j] = evaluate_map(
                map_data, prefix, wing_inputs, height_inputs, with_gurney
            )

    return x_values, y_values, Z


def _title(prefix: str, model_kind: str, with_gurney: bool) -> str:
    suffix = "Gurney" if with_gurney else "NoGurney"
    return f"{prefix}_{model_kind.title()}_Model_{suffix}"


def _label(model: Dict, axis_index: int) -> str:
    name = list(model["variables"].keys())[axis_index]
    unit = model["variables"][name].get("unit", "")
    return f"{name.upper()} ({unit})" if unit else name.upper()


def generate_contours_for_map(prefix: str, map_data: Dict, output_dir: Path, num_points: int) -> Path:
    """Crea una figura interactiva con los cuatro contour plots para un mapa concreto."""

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            _title(prefix, "wing", True),
            _title(prefix, "height", True),
            _title(prefix, "wing", False),
            _title(prefix, "height", False),
        ),
    )

    combinations = (
        ("wing", True, 1, 1),
        ("height", True, 1, 2),
        ("wing", False, 2, 1),
        ("height", False, 2, 2),
    )

    datasets: List[Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    iso_trace_indices: List[int] = []
    for model_kind, with_gurney, row, col in combinations:
        model = map_data["models"]["with_gurney" if with_gurney else "without_gurney"][
            f"{prefix}_{model_kind}_model"
        ]
        x_values, y_values, Z = _grid_for_model(
            prefix, map_data, model_kind, with_gurney, num_points
        )
        datasets.append((model, x_values, y_values, Z, row, col))

    all_values = [Z for (_, _, _, Z, _, _) in datasets]
    z_min = min(float(Z.min()) for Z in all_values)
    z_max = max(float(Z.max()) for Z in all_values)

    for model, x_values, y_values, Z, row, col in datasets:
        contour = go.Contour(
            x=x_values,
            y=y_values,
            z=Z,
            coloraxis="coloraxis",
            contours=dict(showlines=False),
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.3f}<extra></extra>",
        )
        figure.add_trace(contour, row=row, col=col)
        figure.update_xaxes(title_text=_label(model, 0), row=row, col=col)
        figure.update_yaxes(title_text=_label(model, 1), row=row, col=col)

        iso_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            line=dict(color="red"),
            marker=dict(color="red", size=6),
            name="Nivel solicitado",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{text}<extra></extra>",
            showlegend=False,
        )
        figure.add_trace(iso_trace, row=row, col=col)
        iso_trace_indices.append(len(figure.data) - 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_contours.html"

    figure.update_layout(
        title_text=f"Modelos {prefix}",
        coloraxis=dict(colorscale="Viridis", cmin=z_min, cmax=z_max, colorbar=dict(title=prefix)),
    )

    div_id = f"{prefix.lower()}-contours"
    contour_payload = [
        {
            "title": _title(prefix, model_kind, with_gurney),
            "x": x.tolist(),
            "y": y.tolist(),
            "z": Z.tolist(),
        }
        for (model, x, y, Z, _row, _col), (model_kind, with_gurney, *_rest) in zip(
            datasets, combinations
        )
    ]
    iso_payload = {
        "divId": div_id,
        "prefix": prefix,
        "isoTraceIndices": iso_trace_indices,
        "datasets": contour_payload,
    }

    template = Template(
        textwrap.dedent(
            """
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.alignItems = "center";
            container.style.gap = "8px";
            container.style.margin = "10px 0";

            const label = document.createElement("label");
            label.textContent = "Nivel objetivo ($prefix):";
            label.htmlFor = "iso-input";

            const input = document.createElement("input");
            input.id = "iso-input";
            input.type = "number";
            input.step = "any";
            input.placeholder = "Introduce valor";

            const button = document.createElement("button");
            button.textContent = "Buscar puntos";

            const status = document.createElement("div");
            status.style.marginLeft = "12px";
            status.style.fontStyle = "italic";

            container.appendChild(label);
            container.appendChild(input);
            container.appendChild(button);
            container.appendChild(status);

            const plotElement = document.getElementById("$div_id");
            plotElement.parentNode.insertBefore(container, plotElement);

            const isoInfo = $iso_payload;
            const plotDiv = document.getElementById(isoInfo.divId);
            const tolerance = 1e-6;

            function pointsForValue(dataset, target) {
              const matches = [];
              const yLength = dataset.y.length;
              const xLength = dataset.x.length;
              for (let i = 0; i < yLength; i++) {
                for (let j = 0; j < xLength; j++) {
                  const value = dataset.z[i][j];
                  if (Math.abs(value - target) <= tolerance) {
                    matches.push([dataset.x[j], dataset.y[i], value]);
                  }
                }
              }
              return matches.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
            }

            function updateIsoLines(target) {
              let totalPoints = 0;
              isoInfo.datasets.forEach((dataset, idx) => {
                const points = pointsForValue(dataset, target);
                totalPoints += points.length;
                const xs = points.map((p) => p[0]);
                const ys = points.map((p) => p[1]);
                const zs = points.map((p) => p[2].toFixed(3));

                Plotly.restyle(plotDiv, {
                  x: [xs],
                  y: [ys],
                  text: [zs],
                  mode: points.length > 1 ? "lines+markers" : "markers",
                }, isoInfo.isoTraceIndices[idx]);
              });

              if (!Number.isFinite(target)) {
                status.textContent = "Introduce un valor numérico válido.";
              } else if (totalPoints === 0) {
                status.textContent = "No se encontraron puntos enteros que cumplan el nivel.";
              } else {
                status.textContent = `Se encontraron $${totalPoints} puntos que cumplen el nivel.`;
              }
            }

            button.addEventListener("click", () => {
              const target = parseFloat(input.value);
              updateIsoLines(target);
            });

            input.addEventListener("keypress", (event) => {
              if (event.key === "Enter") {
                const target = parseFloat(input.value);
                updateIsoLines(target);
              }
            });
            """
        )
    )
    post_script = template.substitute(
        prefix=prefix, div_id=div_id, iso_payload=json.dumps(iso_payload)
    )

    html = figure.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        div_id=div_id,
        post_script=post_script,
    )
    output_path.write_text(html, encoding="utf-8")

    return output_path


def load_map_file(filename: Path) -> Dict:
    with filename.open("r", encoding="utf-8") as aero_file:
        return json.load(aero_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera contour plots para los modelos de los aero mapas.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        help="Número de puntos por eje usados para muestrear cada superficie (default: 75)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directorio donde guardar las figuras generadas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = []
    for prefix, filename in MAP_SPECS:
        map_data = load_map_file(AERO_DIR / filename)
        saved_paths.append(
            generate_contours_for_map(prefix, map_data, args.output_dir, args.points)
        )

    print("Figuras generadas:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()