from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Evaluation:
    model: Pipeline
    features: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: pd.Series
    class_labels: list[Any]
    accuracy: float
    macro_f1: float


def validate_dataset(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in data.columns:
        raise ValueError("Selecciona una columna objetivo que exista en el CSV.")
    if len(data.columns) < 2:
        raise ValueError("El conjunto debe incluir al menos una variable y una columna objetivo.")

    cleaned = data.dropna(subset=[target]).copy()
    if cleaned.empty:
        raise ValueError("La columna objetivo no contiene valores utilizables.")
    labels = cleaned[target]
    if labels.nunique() != 2:
        raise ValueError(
            "KNN está configurado como clasificador binario: la columna objetivo debe tener "
            "exactamente dos clases distintas."
        )
    if labels.value_counts().min() < 2:
        raise ValueError("Cada clase necesita al menos dos registros para evaluar el modelo.")

    features = cleaned.drop(columns=[target])
    if features.empty:
        raise ValueError("Incluye al menos una columna de variables además del objetivo.")
    usable_features = features.dropna(axis=1, how="all")
    if usable_features.empty:
        raise ValueError("Las variables no contienen valores utilizables.")
    return usable_features, labels


def evaluate_knn(data: pd.DataFrame, target: str, neighbors: int) -> Evaluation:
    features, labels = validate_dataset(data, target)
    if neighbors < 1:
        raise ValueError("El número de vecinos debe ser al menos 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    if neighbors > len(X_train):
        raise ValueError(
            f"k={neighbors} supera los {len(X_train)} registros disponibles para entrenamiento."
        )

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessing = ColumnTransformer(transformers=transformers)
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", KNeighborsClassifier(n_neighbors=neighbors)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    class_labels = sorted(labels.unique().tolist(), key=str)
    return Evaluation(
        model=model,
        features=features,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        predictions=pd.Series(predictions, index=y_test.index, name="prediccion"),
        class_labels=class_labels,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    )


def predict_one(result: Evaluation, values: dict[str, Any]) -> Any:
    missing_columns = [column for column in result.features.columns if column not in values]
    if missing_columns:
        raise ValueError(f"Faltan valores para estas variables: {', '.join(missing_columns)}.")
    sample = pd.DataFrame([{column: values[column] for column in result.features.columns}])
    return result.model.predict(sample)[0]
