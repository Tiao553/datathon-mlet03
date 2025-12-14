import logging
import os
from typing import Optional

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 1. Console Handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Logstash Handler (if configured)
        logstash_host = os.getenv("LOGSTASH_HOST")
        logstash_port = os.getenv("LOGSTASH_PORT")
        
        if logstash_host and logstash_port:
            try:
                from logstash_async.handler import AsynchronousLogstashHandler
                
                ls_handler = AsynchronousLogstashHandler(
                    host=logstash_host, 
                    port=int(logstash_port), 
                    database_path=None
                )
                logger.addHandler(ls_handler)
                # Avoid propagating to root logger to prevent double logging if not handled right
                logger.propagate = False 
            except ImportError:
                print("python-logstash-async not installed. Skipping logstash handler.")
            except Exception as e:
                print(f"Failed to initialize Logstash handler: {e}")

    return logger
