# orchestrator/pipeline_orchestrator.py

import logging

from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.eda_agent import EDAAgent
from agents.modeling_agent import ModelingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.preprocessing_llm_agent import PreprocessingLLMAgent
from agents.modeling_llm_agent import ModelingLLMAgent
from agents.reporting_llm_agent import ReportingLLMAgent

from pipeline_context import PipelineContext

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        logger.info("Initializing agents...")

        # Agents
        self.ingestion_agent = IngestionAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.eda_agent = EDAAgent()
        self.modeling_agent = ModelingAgent()
        self.evaluation_agent = EvaluationAgent()

        self.preprocessing_llm_agent = PreprocessingLLMAgent()
        self.modeling_llm_agent = ModelingLLMAgent()
        self.reporting_llm_agent = ReportingLLMAgent()

        # Shared context
        self.context = PipelineContext()

    def run_pipeline(self):
        logger.info("🚀 Starting Data Analysis Pipeline...")

        # 1️⃣ Ingestion
        data = self.ingestion_agent.run()
        self.context.dataset_sample = data

        # 2️⃣ Preprocessing
        preprocessed_data = self.preprocessing_agent.run(data)

        # 2️⃣a LLM Preprocessing Suggestion
        self.context.preprocessing_suggestion = self.preprocessing_llm_agent.suggest_preprocessing(preprocessed_data)
        print("\n📢 LLM Preprocessing Suggestion:\n")
        print(self.context.preprocessing_suggestion)

        # 3️⃣ EDA
        self.eda_agent.run(preprocessed_data)

        # 3️⃣a LLM Modeling Suggestion
        target_column = "target"  # You can make this dynamic later
        self.context.modeling_suggestion = self.modeling_llm_agent.suggest_modeling_strategy(preprocessed_data, target_column)
        print("\n📢 LLM Modeling Suggestion:\n")
        print(self.context.modeling_suggestion)

        # 4️⃣ Modeling
        model, X_test, y_test = self.modeling_agent.run(preprocessed_data)

        # 5️⃣ Evaluation
        self.evaluation_agent.run(model, X_test, y_test)
        # For now we can hard-code or parse the evaluation report later
        self.context.evaluation_summary = "Accuracy, Precision, Recall, F1 scores calculated. See evaluation report."

        # 6️⃣ LLM Reporting
        report = self.reporting_llm_agent.generate_report(
            self.context.dataset_sample,
            self.context.preprocessing_suggestion,
            self.context.modeling_suggestion,
            self.context.evaluation_summary
        )
        self.context.final_report = report

        print("\n📢 LLM Pipeline Report:\n")
        print(report)

        logger.info("✅ Pipeline completed.")
