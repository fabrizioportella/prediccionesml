from __future__ import annotations

import numpy as np
import pandas as pd


def _satisfaction_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 360
    response_time = rng.integers(1, 49, size)
    wait_time = rng.integers(1, 36, size)
    resolved = rng.choice(["Sí", "No"], size=size, p=[0.78, 0.22])
    communication = rng.integers(1, 6, size)
    service_quality = rng.integers(1, 6, size)
    age = rng.integers(18, 71, size)
    score = (
        0.42 * service_quality
        + 0.36 * communication
        + 1.25 * (resolved == "Sí")
        - 0.035 * response_time
        - 0.028 * wait_time
        + rng.normal(0, 0.8, size)
    )
    satisfied = np.where(score >= np.median(score), "Satisfecho", "No satisfecho")
    return pd.DataFrame(
        {
            "edad": age,
            "tiempo_espera_min": wait_time,
            "tiempo_respuesta_h": response_time,
            "problema_resuelto": resolved,
            "calidad_servicio_1_5": service_quality,
            "comunicacion_1_5": communication,
            "satisfaccion": satisfied,
        }
    )


def _investment_data() -> pd.DataFrame:
    rng = np.random.default_rng(73)
    size = 420
    age = rng.integers(21, 71, size)
    capital_available = rng.integers(500, 60_001, size)
    emergency_fund_months = rng.integers(0, 13, size)
    investment_horizon_years = rng.integers(1, 16, size)
    debt_ratio = rng.uniform(0, 0.9, size).round(2)
    risk_tolerance = rng.integers(1, 6, size)
    experience_years = rng.integers(0, 16, size)
    score = (
        0.23 * emergency_fund_months
        + 0.21 * investment_horizon_years
        + 0.42 * risk_tolerance
        + 0.13 * experience_years
        - 5.2 * debt_ratio
        + 0.000025 * capital_available
        + rng.normal(0, 1.1, size)
    )
    decision = np.where(score >= np.median(score), "Sí", "No")
    return pd.DataFrame(
        {
            "edad": age,
            "capital_disponible": capital_available,
            "fondo_emergencia_meses": emergency_fund_months,
            "horizonte_inversion_anios": investment_horizon_years,
            "proporcion_deuda": debt_ratio,
            "tolerancia_riesgo_1_5": risk_tolerance,
            "experiencia_inversion_anios": experience_years,
            "deberia_invertir": decision,
        }
    )


DEMO_DATASETS = {
    "satisfaction": {"data": _satisfaction_data(), "target": "satisfaccion"},
    "investment": {"data": _investment_data(), "target": "deberia_invertir"},
}
