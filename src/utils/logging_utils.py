import os
import logging
import transformers

def setup_logging_to_file(log_dir: str) -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=f"{log_dir}/output.log",
        filemode='a',
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s',  
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_propagation()

    # Set specific loggers to ERROR level, to avoid spamming the logs
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

