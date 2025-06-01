# agents/modeling_llm_agent.py

from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class ModelingLLMAgent:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q4_0.gguf"):
        logger.info("Loading LLaMA model for ModelingLLMAgent...")
        self.llm = Llama(model_path=model_path)

    def suggest_modeling_strategy(self, df_sample, target_column):
        logger.info("Generating modeling strategy suggestions using LLM...")

        # Build prompt
        prompt = f"""
        ### Instruction:
        You are a senior data scientist tasked with designing a modeling pipeline.
        The dataset sample is:

        {df_sample.head(5).to_markdown()}

        The target column is: {target_column}

        Please suggest:
        - Which model(s) to try (e.g. Decision Tree, RandomForest, XGBoost, etc)
        - How to tune hyperparameters
        - How to do feature selection or engineering
        - How to validate the model

        ### Response:
        """

        # Generate response
        output = self.llm(prompt, max_tokens=512)
        response = output["choices"][0]["text"].strip()

        logger.info("LLM modeling suggestion complete.")
        return response
