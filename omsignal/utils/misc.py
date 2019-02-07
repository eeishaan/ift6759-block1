#!/usr/bin/env python3

import logging
import os

logger = logging.getLogger(__name__)


def check_file(file_path, default_path):
    if not os.path.isfile(file_path):
        message = 'Unable to find file at {}. Trying at default location.'\
            .format(file_path)
        logger.info(message)
        file_path = default_path / file_path
        if not os.path.isfile(file_path):
            message = 'Unable to find file at {}'\
                .format(file_path)
            logger.error(message)
            return None
    return file_path