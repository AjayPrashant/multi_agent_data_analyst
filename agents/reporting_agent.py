# agents/reporting_agent.py

import os
import logging

logger = logging.getLogger(__name__)


class ReportingAgent:
    def __init__(self, reports_folder='outputs/reports'):
        self.reports_folder = reports_folder

    def run(self):
        logger.info("📄 Reporting Agent: Generating final report summary...")

        eda_report_path = os.path.join(self.reports_folder, "eda_report.html")
        eval_report_path = os.path.join(self.reports_folder, "evaluation_report.txt")

        logger.info("Final Report Summary:")
        logger.info(f"- EDA Report: {eda_report_path}")
        logger.info(f"- Evaluation Report: {eval_report_path}")
