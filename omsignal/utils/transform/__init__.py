from abc import ABC, abstractmethod

class OmTransform(ABC):
    '''
    Abstract base class for custom transformers
    '''

    @abstractmethod
    def state(self):
        '''
        This method should return the state of the transformer
        '''
        pass