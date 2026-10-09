"""
Visualización y reporte: heatmap, volcano plot y resumen.

Genera figuras analíticas como SVG autónomo (sin dependencias de
matplotlib en el servidor) y un resumen estructurado (JSON + Markdown)
que se puede descargar desde la UI.

- ``heatmap_svg``: mapa de calor de una matriz (p. ej. expresión o
  genotipos), con reescalado por color.
- ``volcano_svg``: log2(fold-change) vs -log10(p) con umbrales.
- ``build_report``: resumen Markdown/JSON a partir de los resultados de
  los hitos anteriores (QC, k-mers, mapeo, variantes, anotación).
"""

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Paleta del heatmap (azul -> blanco -> rojo), en formato hex.
_HEATMAP_COLORS = [
    (13, 30, 60), (30, 80, 150), (80, 150, 210), (200, 225, 240),
    (245, 245, 235), (250, 210, 160), (235, 140, 90), (190, 50, 40),
    (110, 15, 20),
]


def _lerp_color(t: float) -> str:
    """Color interpolado de la paleta para t en [0, 1]."""
    t = max(0.0, min(1.0, t))
    pos = t * (len(_HEATMAP_COLORS) - 1)
    i = int(pos)
    frac = pos - i
    if i >= len(_HEATMAP_COLORS) - 1:
        r, g, b = _HEATMAP_COLORS[-1]
    else:
        r0, g0, b0 = _HEATMAP_COLORS[i]
        r1, g1, b1 = _HEATMAP_COLORS[i + 1]
        r = int(r0 + (r1 - r0) * frac)
        g = int(g0 + (g1 - g0) * frac)
        b = int(b0 + (b1 - b0) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def heatmap_svg(matrix: Sequence[Sequence[float]],
                row_labels: Optional[Sequence[str]] = None,
                col_labels: Optional[Sequence[str]] = None,
                title: str = "Heatmap",
                max_cells: int = 4000,
                cell_size: int = 18) -> str:
    """
    Genera un heatmap SVG de una matriz (valores reescalados por columnas
    a z-scores y mapeados a color). Si la matriz tiene más de ``max_cells``
    celdas, se submuestrea por filas/columnas para acotar el SVG.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"

    n, p = M.shape
    if n * p > max_cells:
        step_r = max(1, int(np.ceil(n / int(np.sqrt(max_cells)))))
        step_c = max(1, int(np.ceil(p / int(np.sqrt(max_cells)))))
        M = M[::step_r, ::step_c]
        n, p = M.shape
        if row_labels:
            row_labels = list(row_labels)[::step_r]
        if col_labels:
            col_labels = list(col_labels)[::step_c]

    # z-score por columna (evita que rasgos con escalas distintas dominen)
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.where(sd > 1e-9, sd, 1.0)
    vmax = float(np.abs(Z).max()) or 1.0
    T = (Z / vmax + 1.0) / 2.0  # [0,1]

    margin_l = 90 if row_labels else 10
    margin_t = 34 if title else 8
    margin_b = 60 if col_labels else 10
    width = margin_l + p * cell_size + 10
    height = margin_t + n * cell_size + margin_b

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
             f"height='{height}' viewBox='0 0 {width} {height}'>",
             "<rect width='100%' height='100%' fill='#0b1220'/>"]
    if title:
        parts.append(f"<text x='{width/2:.0f}' y='20' fill='#c8ffec' "
                     f"font-family='monospace' font-size='13' "
                     f"text-anchor='middle'>{html.escape(title)}</text>")

    for i in range(n):
        for j in range(p):
            parts.append(
                f"<rect x='{margin_l + j*cell_size}' y='{margin_t + i*cell_size}' "
                f"width='{cell_size}' height='{cell_size}' "
                f"fill='{_lerp_color(float(T[i, j]))}'/>")

    if row_labels:
        for i, lab in enumerate(row_labels[:n]):
            parts.append(
                f"<text x='{margin_l - 6}' y='{margin_t + i*cell_size + cell_size*0.7:.0f}' "
                f"fill='#8fa3b0' font-family='monospace' font-size='9' "
                f"text-anchor='end'>{html.escape(str(lab))}</text>")
    if col_labels:
        for j, lab in enumerate(col_labels[:p]):
            x = margin_l + j * cell_size + cell_size * 0.5
            parts.append(
                f"<text x='{x:.0f}' y='{margin_t + n*cell_size + 12}' "
                f"fill='#8fa3b0' font-family='monospace' font-size='9' "
                f"text-anchor='end' transform='rotate(-60 {x:.0f} "
                f"{margin_t + n*cell_size + 12})'>{html.escape(str(lab))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def volcano_svg(fold_changes: Sequence[float], p_values: Sequence[float],
                labels: Optional[Sequence[str]] = None,
                fc_threshold: float = 1.0,
                p_threshold: float = 0.05,
                title: str = "Volcano plot") -> str:
    """
    Volcano plot SVG: log2(FC) en x, -log10(p) en y. Puntos significativos
    (|log2FC| >= umbral y p <= umbral) se colorean en acento.
    """
    fc = np.asarray(fold_changes, dtype=float)
    pv = np.asarray(p_values, dtype=float)
    m = min(len(fc), len(pv))
    fc, pv = fc[:m], pv[:m]
    if m == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    pv = np.clip(pv, 1e-300, 1.0)
    y = -np.log10(pv)

    W, H = 640, 440
    PL, PR, PT, PB = 46, 14, 34, 40
    PW, PH = W - PL - PR, H - PT - PB
    xmax = max(float(np.abs(fc).max()), fc_threshold * 1.2) or 1.0
    ymax = max(float(y.max()), -np.log10(p_threshold) * 1.2) or 1.0
    def X(v):
        return PL + ((v + xmax) / (2 * xmax)) * PW

    def Y(v):
        return PT + (1 - v / ymax) * PH

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
             f"viewBox='0 0 {W} {H}'>",
             "<rect width='100%' height='100%' fill='#0b1220'/>",
             f"<text x='{W/2:.0f}' y='20' fill='#c8ffec' font-family='monospace' "
             f"font-size='13' text-anchor='middle'>{html.escape(title)}</text>"]
    # ejes y umbrales
    parts.append(f"<line x1='{PL}' y1='{PT+PH}' x2='{W-PR}' y2='{PT+PH}' stroke='#3a4a58'/>")
    parts.append(f"<line x1='{PL}' y1='{PT}' x2='{PL}' y2='{PT+PH}' stroke='#3a4a58'/>")
    parts.append(f"<line x1='{X(fc_threshold)}' y1='{PT}' x2='{X(fc_threshold)}' "
                 f"y2='{PT+PH}' stroke='#5a6a78' stroke-dasharray='3 3'/>")
    parts.append(f"<line x1='{X(-fc_threshold)}' y1='{PT}' x2='{X(-fc_threshold)}' "
                 f"y2='{PT+PH}' stroke='#5a6a78' stroke-dasharray='3 3'/>")
    parts.append(f"<line x1='{PL}' y1='{Y(-np.log10(p_threshold))}' x2='{W-PR}' "
                 f"y2='{Y(-np.log10(p_threshold))}' stroke='#5a6a78' stroke-dasharray='3 3'/>")

    for i in range(m):
        sig = abs(fc[i]) >= fc_threshold and pv[i] <= p_threshold
        color = "#40e0b2" if sig else "#5a6a78"
        r = 3.2 if sig else 2.2
        parts.append(f"<circle cx='{X(fc[i]):.1f}' cy='{Y(y[i]):.1f}' r='{r}' "
                     f"fill='{color}' fill-opacity='0.85'/>")
    parts.append(f"<text x='{PL+PW/2:.0f}' y='{H-8}' fill='#8fa3b0' "
                 f"font-family='monospace' font-size='10' text-anchor='middle'>"
                 f"log2 fold-change</text>")
    parts.append(f"<text x='14' y='{PT+PH/2:.0f}' fill='#8fa3b0' "
                 f"font-family='monospace' font-size='10' text-anchor='middle' "
                 f"transform='rotate(-90 14 {PT+PH/2:.0f})'>-log10(p)</text>")
    parts.append("</svg>")
    return "".join(parts)


def manhattan_svg(positions: Sequence[float], p_values: Sequence[float],
                  sig_threshold: float = 5e-8,
                  title: str = "Manhattan plot") -> str:
    """
    Manhattan plot SVG: posición/índice en x, -log10(p) en y. Los puntos
    por encima de la línea de significancia se resaltan en acento.
    """
    pos = np.asarray(positions, dtype=float)
    pv = np.asarray(p_values, dtype=float)
    m = min(len(pos), len(pv))
    pos, pv = pos[:m], pv[:m]
    if m == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    pv = np.clip(pv, 1e-300, 1.0)
    y = -np.log10(pv)

    W, H = 680, 360
    PL, PR, PT, PB = 46, 14, 34, 40
    PW, PH = W - PL - PR, H - PT - PB
    xmin, xmax = float(pos.min()), float(pos.max())
    xspan = (xmax - xmin) or 1.0
    ymax = max(float(y.max()), -np.log10(sig_threshold) * 1.2) or 1.0
    def X(v):
        return PL + ((v - xmin) / xspan) * PW

    def Y(v):
        return PT + (1 - v / ymax) * PH
    sig_line = -np.log10(sig_threshold)

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
             f"viewBox='0 0 {W} {H}'>",
             "<rect width='100%' height='100%' fill='#0b1220'/>",
             f"<text x='{W/2:.0f}' y='20' fill='#c8ffec' font-family='monospace' "
             f"font-size='13' text-anchor='middle'>{html.escape(title)}</text>"]
    parts.append(f"<line x1='{PL}' y1='{PT+PH}' x2='{W-PR}' y2='{PT+PH}' stroke='#3a4a58'/>")
    parts.append(f"<line x1='{PL}' y1='{PT}' x2='{PL}' y2='{PT+PH}' stroke='#3a4a58'/>")
    parts.append(f"<line x1='{PL}' y1='{Y(sig_line):.1f}' x2='{W-PR}' "
                 f"y2='{Y(sig_line):.1f}' stroke='#f87171' stroke-dasharray='3 3'/>")

    for i in range(m):
        sig = pv[i] <= sig_threshold
        color = "#40e0b2" if sig else "#5a6a78"
        r = 3.4 if sig else 2.2
        parts.append(f"<circle cx='{X(pos[i]):.1f}' cy='{Y(y[i]):.1f}' r='{r}' "
                     f"fill='{color}' fill-opacity='0.85'/>")
    parts.append(f"<text x='{PL+PW/2:.0f}' y='{H-8}' fill='#8fa3b0' "
                 f"font-family='monospace' font-size='10' text-anchor='middle'>"
                 f"marcador</text>")
    parts.append(f"<text x='14' y='{PT+PH/2:.0f}' fill='#8fa3b0' "
                 f"font-family='monospace' font-size='10' text-anchor='middle' "
                 f"transform='rotate(-90 14 {PT+PH/2:.0f})'>-log10(p)</text>")
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class Report:
    """Resumen estructurado de un análisis."""
    title: str
    sections: List[dict]
    markdown: str
    json: dict


def build_report(title: str, results: Dict[str, dict]) -> Report:
    """
    Construye un reporte Markdown/JSON a partir de resultados de análisis.

    Args:
        title: Título del reporte.
        results: Dict ``{seccion: {clave: valor}}``; los valores se
            tabulan como listas de métricas.

    Returns:
        :class:`Report` con ``markdown``, ``json`` y ``sections``.
    """
    sections: List[dict] = []
    md = [f"# {title}", ""]
    for name, data in results.items():
        md.append(f"## {name}")
        md.append("")
        rows = []
        for key, value in (data or {}).items():
            if isinstance(value, float):
                value = round(value, 4)
            rows.append({"metric": key, "value": value})
            md.append(f"- **{key}**: {value}")
        md.append("")
        sections.append({"name": name, "rows": rows})
    return Report(title=title, sections=sections,
                  markdown="\n".join(md), json={"title": title,
                                                "sections": sections})