from abc import ABC, abstractmethod


class MDPEnvironment[State, Action](ABC):
    @abstractmethod
    def transitions(
        self, state: State, action: Action
    ) -> list[tuple[float, State, float]]: ...
