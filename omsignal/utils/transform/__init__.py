#!/usr/bin/env python3
from abc import ABC, abstractmethod

class OmTransform(ABC):
    '''
    Abstract base class for custom transformers
    '''

    @abstractmethod
    def __call__(self, x):
        pass


    def state(self):
        '''
        This method should return the state of the transformer
        '''
        pass