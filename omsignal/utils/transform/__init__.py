#!/usr/bin/env python3
from abc import ABC

class OmTransform(ABC):
    '''
    Abstract base class for custom transformers
    '''

    def state(self):
        '''
        This method should return the state of the transformer
        '''
        pass