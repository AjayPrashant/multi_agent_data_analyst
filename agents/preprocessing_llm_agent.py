# agents/preprocessing_llm_agent.py

import os
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class PreprocessingLLMAgent:
    def __init__(self):
        logger.info("Loading local Mistral 7B v0.3 model with llama-cpp-python...")

        # Path to your local GGUF model
        model_path = "/Users/ajayprashantmuralidharan/Agentic Projects/mistral-7b-instruct-v0.3.Q4_0.gguf"

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=True
        )

    def suggest_preprocessing(self, df_sample):
        logger.info("Generating preprocessing suggestions with local model...")

        prompt = f"""
        ### Instruction:
        You are an expert data scientist helping preprocess a dataset.
        Here is a sample of the dataset:

        {df_sample.head(3).to_markdown()}

        Please suggest preprocessing steps.

        ### Response:
        """

        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["###"]
        )

        text_output = response["choices"][0]["text"].strip()
        logger.info("LLM preprocessing suggestion complete.")
        return text_output
