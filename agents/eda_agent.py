# agents/eda_agent.py

import pandas as pd
from ydata_profiling import ProfileReport
import os
import logging

logger = logging.getLogger(__name__)


class EDAAgent:
    def __init__(self, output_folder='outputs/reports'):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self, data: pd.DataFrame):
        logger.info("📊 EDA Agent: Generating EDA report...")

        profile = ProfileReport(data, title="EDA Report", explorative=True)
        report_path = os.path.join(self.output_folder, "eda_report.html")

        profile.to_file(report_path)

        logger.info(f"EDA Report saved to: {report_path}")
