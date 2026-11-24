import unittest

import pandas as pd

from knn_app.model import evaluate_knn, predict_one, validate_dataset


class KnnModelTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "monto": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "canal": ["web", "web", "tienda", "tienda"] * 3,
                "resultado": ["No", "No", "Sí", "Sí"] * 3,
            }
        )

    def test_evaluation_and_prediction_support_numeric_and_categorical_features(self):
        result = evaluate_knn(self.data, "resultado", neighbors=3)
        predicted = predict_one(result, {"monto": 4.0, "canal": "nuevo"})

        self.assertIn(predicted, {"Sí", "No"})
        self.assertEqual(len(result.predictions), len(result.y_test))
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_validation_drops_rows_without_target(self):
        data = self.data.copy()
        data.loc[0, "resultado"] = None

        features, labels = validate_dataset(data, "resultado")

        self.assertEqual(len(features), 11)
        self.assertEqual(len(labels), 11)

    def test_validation_rejects_non_binary_target(self):
        data = self.data.copy()
        data.loc[0, "resultado"] = "Tal vez"

        with self.assertRaisesRegex(ValueError, "exactamente dos clases"):
            validate_dataset(data, "resultado")


if __name__ == "__main__":
    unittest.main()
