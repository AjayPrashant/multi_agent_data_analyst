# run_pipeline.py

from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from logging_config import setup_logging

def main():
    setup_logging()

    import logging
    logger = logging.getLogger("Main")
    logger.info("🚀 Starting Data Analysis Pipeline...")

    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline()

if __name__ == "__main__":
    main()
