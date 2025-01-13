from abc import ABC, abstractmethod

class LanggraphNode(ABC):
    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def run(self, state):
        raise NotImplementedError()

    def get_as_kwargs(self):
        return {
            'node': self.get_name(),
            'action': lambda state: self.run(state)
        }