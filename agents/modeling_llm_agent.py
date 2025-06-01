# agents/modeling_llm_agent.py

import os
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ModelingLLMAgent:
    def __init__(self):
        logger.info("Loading local Mistral 7B v0.3 model with llama-cpp-python...")

        model_path = "/Users/ajayprashantmuralidharan/Agentic Projects/mistral-7b-instruct-v0.3.Q4_0.gguf"

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=True
        )

    def suggest_modeling_strategy(self, df_sample, target_column):
        logger.info("Generating modeling strategy with local model...")

        prompt = f"""
        ### Instruction:
        You are a senior data scientist designing a modeling pipeline.
        Dataset sample:

        {df_sample.head(3).to_markdown()}

        Target column: {target_column}

        Please suggest:
        - Model(s) to try
        - Hyperparameter tuning
        - Feature selection ideas
        - Validation strategy

        ### Response:
        """

        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["###"]
        )

        text_output = response["choices"][0]["text"].strip()
        logger.info("LLM modeling suggestion complete.")
        return text_output
