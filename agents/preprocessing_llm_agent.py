# agents/preprocessing_llm_agent.py

from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class PreprocessingLLMAgent:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        logger.info("Loading LLaMA model for PreprocessingLLMAgent...")
        self.llm = Llama(model_path=model_path)

    def suggest_preprocessing(self, df_sample):
        logger.info("Generating preprocessing suggestions using LLM...")

        # Build prompt
        prompt = f"""
        ### Instruction:
        You are an expert data scientist helping preprocess a dataset.
        Here is a sample of the dataset:

        {df_sample.head(5).to_markdown()}

        Please suggest detailed preprocessing steps I should take on this dataset.
        Mention:
        - How to handle missing values
        - Encoding for categorical columns
        - Scaling for numerical columns
        - Any feature engineering ideas

        ### Response:
        """

        # Generate response
        output = self.llm(prompt, max_tokens=512)
        response = output["choices"][0]["text"].strip()

        logger.info("LLM preprocessing suggestion complete.")
        return response
