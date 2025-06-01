# agents/reporting_llm_agent.py

from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class ReportingLLMAgent:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_0.gguf"):
        logger.info("Loading LLaMA model for ReportingLLMAgent...")
        self.llm = Llama(model_path=model_path)

    def generate_report(self, df_sample, preprocessing_summary, modeling_summary, evaluation_summary):
        logger.info("Generating pipeline report using LLM...")

        # Build prompt
        prompt = f"""
        ### Instruction:
        You are a data scientist creating a report for an ML pipeline.
        Here is the context:

        - Preprocessing summary:
        {preprocessing_summary}
        - Modeling summary:
        {modeling_summary}
        - Evaluation summary:
        {evaluation_summary}
        Please generate a clear report that includes:
        - Summary of the dataset
        - Key preprocessing steps taken
        - Model used and why
        - Evaluation results

        ### Response:
        """

        # Generate response
        output = self.llm(prompt, max_tokens=256) #changed from 1024
        response = output["choices"][0]["text"].strip()

        logger.info("LLM reporting complete.")
        return response
