from ..environments.shuttleenvironment import (
    ShuttleEnvironment,
    ShuttleEnvironmentAction,
)
from .mdpsolver import MDPSolver


class ShuttleValueIteration(MDPSolver[int, ShuttleEnvironmentAction]):
    def __init__(
        self,
        env: ShuttleEnvironment,
        max_state: int,
        gamma: float,
        tolerance: float = 1e-8,
        record_history: bool = False,
    ) -> None:
        super().__init__(env)

        # Value iteration only converges for a discount factor strictly below 1
        if not 0 <= gamma < 1:
            raise ValueError("Discount factor gamma must be in [0, 1).")

        self.max_state = max_state
        self.gamma = gamma
        self.tolerance = tolerance
        self.record_history = record_history
        self.V: list[float] = [0.0] * (self.max_state + 1)
        self.deltas: list[float] = []  # Largest value change per sweep
        self.history: list[list[float]] = []  # Copy of V after each sweep

    def q_values(
        self, state: int, V: list[float] | None = None
    ) -> dict[ShuttleEnvironmentAction, float]:
        """Q-values of a state with respect to V (default: the current values)."""
        V = self.V if V is None else V

        # Next states beyond max_state are clamped (customers are turned away)
        return {
            action: sum(
                p * (r + self.gamma * V[min(s_next, self.max_state)])
                for p, s_next, r in self.env.transitions(state, action)
            )
            for action in ShuttleEnvironmentAction
        }

    def solve(self) -> None:
        self.V = [0.0] * (self.max_state + 1)
        self.deltas = []
        self.history = []

        while True:
            delta = 0.0
            for s in range(self.max_state + 1):
                new_value = max(self.q_values(s).values())
                delta = max(delta, abs(new_value - self.V[s]))
                self.V[s] = new_value

            self.deltas.append(delta)
            if self.record_history:
                self.history.append(self.V.copy())
            if delta < self.tolerance:
                break

    def get_value(self, state: int, time_step: int | None = None) -> float:
        return self.V[state]

    def get_action(
        self, state: int, time_step: int | None = None
    ) -> ShuttleEnvironmentAction:
        # The policy is stationary, so the time step is ignored
        q_values = self.q_values(state)
        return max(q_values, key=q_values.get)
