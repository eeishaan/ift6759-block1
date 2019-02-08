#!/usr/bin/env python3

import logging
import os

logger = logging.getLogger(__name__)


def check_file(file_path, default_path):
    '''
    Check if file present in provided path. If not present, look for file
    in default location. If not present anywhere return None else return the
    valid file path.
    '''
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
