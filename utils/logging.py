import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)8s @ %(module)12s: %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "formatter": "default",
            "level": logging.DEBUG,
        }
    },
    "loggers": {
        "app": {
            "handlers": ["file"],
            "level": logging.DEBUG,
            "propagate": False,
        }
    },
}
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("app")
