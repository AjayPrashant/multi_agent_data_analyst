# agents/reporting_llm_agent.py

import os
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ReportingLLMAgent:
    def __init__(self):
        logger.info("Loading local Mistral 7B v0.3 model with llama-cpp-python...")

        model_path = "/Users/ajayprashantmuralidharan/Agentic Projects/mistral-7b-instruct-v0.3.Q4_0.gguf"

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=True
        )

    def generate_report(self, df_sample, preprocessing_summary, modeling_summary, evaluation_summary):
        logger.info("Generating pipeline report with local model...")

        prompt = f"""
        ### Instruction:
        You are an expert data scientist creating a report for a machine learning pipeline.
        Here is the context:

        - Dataset sample:

        {df_sample.head(5).to_markdown()}

        - Preprocessing summary:

        {preprocessing_summary}

        - Modeling summary:

        {modeling_summary}

        - Evaluation summary:

        {evaluation_summary}

        Please generate a clear, professional report that includes:
        - Summary of the dataset
        - Key preprocessing steps taken
        - Model(s) used and why
        - Evaluation results
        - Recommendations for improvement
        - Next steps

        ### Response:
        """


        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            stop=["###"]
        )

        text_output = response["choices"][0]["text"].strip()
        logger.info("LLM reporting complete.")
        return text_output
    
    # If crossing input token limit:
    '''        
        prompt = f"""
        ### Instruction:
        You are an expert data scientist creating a professional report.
        Context:

        Dataset sample:

        {df_sample.head(3).to_markdown()}

        Preprocessing summary:
        {preprocessing_summary}

        Modeling summary:
        {modeling_summary}

        Evaluation summary:
        {evaluation_summary}

        Please generate a clear report with:
        - Dataset summary
        - Preprocessing steps
        - Model(s) used
        - Evaluation results
        - Recommendations
        - Next steps

        ### Response:
        """ '''