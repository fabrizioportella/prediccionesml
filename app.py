from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from knn_app.demo_data import DEMO_DATASETS
from knn_app.model import evaluate_knn, predict_one, validate_dataset


st.set_page_config(
    page_title="Nexo | Predicción KNN",
    page_icon="✳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #17231f; --muted: #74817b; --green: #2d7659; --line: #e7ece8; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: #f7f9f7; }
    .block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: var(--ink); }
    h1 { font-size: 2.3rem !important; }
    .brand { color: var(--green); font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 800; }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .subtle { color: var(--muted); font-size: 1rem; }
    .step-title { margin: 1.25rem 0 .25rem; font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 9px; font-weight: 600; min-height: 2.8rem; }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.2rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

FIELD_LABELS = {
    "edad": ("Edad", "Años"),
    "tiempo_espera_min": ("Tiempo de espera", "Minutos"),
    "tiempo_respuesta_h": ("Tiempo de respuesta", "Horas"),
    "problema_resuelto": ("¿Se resolvió el problema?", ""),
    "calidad_servicio_1_5": ("Calidad del servicio", "Escala de 1 a 5"),
    "comunicacion_1_5": ("Calidad de la comunicación", "Escala de 1 a 5"),
    "capital_disponible": ("Capital disponible", "Unidades monetarias"),
    "fondo_emergencia_meses": ("Fondo de emergencia", "Meses cubiertos"),
    "horizonte_inversion_anios": ("Plazo de inversión", "Años"),
    "proporcion_deuda": ("Proporción de deuda", "De 0 a 1"),
    "tolerancia_riesgo_1_5": ("Tolerancia al riesgo", "Escala de 1 a 5"),
    "experiencia_inversion_anios": ("Experiencia invirtiendo", "Años"),
}


def load_csv(uploaded_file: object) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"No se pudo leer el CSV: {error}")
        st.stop()


st.markdown('<p class="brand">✳ Nexo</p>', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">PREDICCIONES CON KNN</p>', unsafe_allow_html=True)
st.title("Prueba una predicción")
st.markdown(
    '<p class="subtle">Elige un tema, completa un caso de ejemplo y mira qué resultado estima el modelo.</p>',
    unsafe_allow_html=True,
)

model_name = st.radio(
    "¿Qué quieres predecir?",
    ("Satisfacción del cliente", "Decisión de inversión"),
    horizontal=True,
)
model_key = "satisfaction" if model_name == "Satisfacción del cliente" else "investment"
demo_data = DEMO_DATASETS[model_key]
data = demo_data["data"].copy()
target = demo_data["target"]
is_demo = True

with st.expander("¿Quieres probar con tu propio archivo CSV?"):
    uploaded_file = st.file_uploader(
        "Sube tus datos (opcional)",
        type=["csv"],
        help="El archivo debe tener encabezados y una columna con dos resultados posibles, por ejemplo Sí y No.",
    )
    if uploaded_file is not None:
        data = load_csv(uploaded_file)
        if data.empty or len(data.columns) < 2:
            st.error("El CSV debe tener filas y al menos dos columnas.")
            st.stop()
        default_target_index = (
            data.columns.get_loc(target) if target in data.columns else len(data.columns) - 1
        )
        target = st.selectbox("¿Qué columna quieres predecir?", data.columns, index=default_target_index)
        is_demo = False

if model_key == "investment":
    st.info(
        "El ejemplo de inversión usa datos inventados y es solo educativo. "
        "No es asesoría financiera ni una recomendación real."
    )
elif is_demo:
    st.caption("Estás usando datos sintéticos de demostración; no son registros de clientes reales.")

try:
    _, target_values = validate_dataset(data, target)
except ValueError as error:
    st.error(str(error))
    st.stop()

max_neighbors = max(1, int(len(target_values) * 0.75))
neighbors = min(5, max_neighbors)
with st.expander("Ajustes del modelo (opcional)"):
    neighbors = st.slider(
        "Casos parecidos que se compararán",
        min_value=1,
        max_value=max_neighbors,
        value=neighbors,
        help="El modelo compara tu caso con los registros más parecidos y elige el resultado más común.",
    )
    if is_demo:
        st.caption(f"El ejemplo contiene {len(data)} casos.")

try:
    result = evaluate_knn(data, target, neighbors)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.markdown('<p class="step-title">Completa los datos del caso</p>', unsafe_allow_html=True)
st.caption("Usamos valores de referencia como punto de partida. Puedes cambiarlos antes de predecir.")

with st.form("prediction_form"):
    values: dict[str, object] = {}
    feature_columns = list(result.features.columns)
    form_columns = st.columns(2)
    for index, column in enumerate(feature_columns):
        series = result.features[column]
        label, unit = FIELD_LABELS.get(
            column,
            (column.replace("_", " ").capitalize(), ""),
        )
        help_text = f"Unidad: {unit}." if unit else None
        with form_columns[index % 2]:
            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                median = float(numeric_series.median()) if not numeric_series.empty else 0.0
                values[column] = st.number_input(
                    label,
                    value=median,
                    help=help_text,
                    key=f"{model_key}_{column}",
                )
            else:
                options = sorted(series.dropna().unique().tolist(), key=str)
                if not options:
                    options = ["Sin dato"]
                values[column] = st.selectbox(
                    label,
                    options,
                    format_func=str,
                    help=help_text,
                    key=f"{model_key}_{column}",
                )
    submitted = st.form_submit_button("Ver predicción", type="primary", use_container_width=True)

if submitted:
    try:
        predicted_class = predict_one(result, values)
        st.success(f"### Resultado estimado: {predicted_class}")
        st.caption("Es una estimación basada en casos parecidos; no garantiza el resultado real.")
    except ValueError as error:
        st.error(str(error))

with st.expander("¿Qué tan bien funciona el modelo?"):
    st.caption(
        f"Separamos los datos para probar el modelo: entrenó con {len(result.X_train)} casos "
        f"y se evaluó con otros {len(result.X_test)} que no vio durante el entrenamiento."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Aciertos en la prueba", f"{result.accuracy:.1%}")
    metric_columns[1].metric("Equilibrio entre resultados", f"{result.macro_f1:.1%}")
    metric_columns[2].metric("Casos de prueba", f"{len(result.y_test):,}")
    figure, axis = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        result.y_test,
        result.predictions,
        labels=result.class_labels,
        display_labels=result.class_labels,
        cmap="Greens",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Resultados reales y estimados", loc="left", fontsize=13, pad=14)
    axis.set_xlabel("Estimación del modelo")
    axis.set_ylabel("Resultado real")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with st.expander("¿Cómo funciona KNN?"):
    st.markdown(
        """
        **KNN busca casos parecidos.** Compara los datos que ingresaste con registros anteriores y
        estima el resultado que aparece con más frecuencia entre ellos. El ajuste **k** define
        cuántos casos cercanos comparar.

        Los ejemplos de esta aplicación son sintéticos. Para obtener resultados útiles con tus
        datos, revisa su calidad y valida las predicciones con conocimiento del tema.
        """
    )
