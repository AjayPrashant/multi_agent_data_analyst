# agents/evaluation_agent.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import logging

logger = logging.getLogger(__name__)


class EvaluationAgent:
    def __init__(self, output_folder='outputs/reports'):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self, model, X_test, y_test):
        logger.info("📝 Evaluation Agent: Evaluating model...")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        report = f"""
        MODEL EVALUATION REPORT
        -----------------------
        Accuracy : {accuracy:.4f}
        Precision: {precision:.4f}
        Recall   : {recall:.4f}
        F1 Score : {f1:.4f}
        """

        report_path = os.path.join(self.output_folder, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Evaluation report saved to: {report_path}")
        logger.info(report)
