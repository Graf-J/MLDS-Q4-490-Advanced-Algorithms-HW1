from ..environments.shuttleenvironment import (
    ShuttleEnvironment,
    ShuttleEnvironmentAction,
)
from .mdpsolver import MDPSolver


class ShuttleEnumeration(MDPSolver[int, ShuttleEnvironmentAction]):
    def __init__(
        self,
        env: ShuttleEnvironment,
        T: int,
        max_initial_customers_waiting: int,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.T = T
        self.max_initial_customers_waiting = max_initial_customers_waiting
        self.gamma = gamma
        self.V: list[list[float]] = []

    def solve(self) -> None:
        max_arrivals = max(self.env.customer_arriving_options)

        self.V = [
            [0.0] * (self.max_initial_customers_waiting + t * max_arrivals + 1)
            for t in range(self.T + 1)
        ]

        for t in reversed(range(self.T)):
            for s in range(len(self.V[t])):
                q_values = {}

                for action in (
                    ShuttleEnvironmentAction.DISPATCH,
                    ShuttleEnvironmentAction.WAIT,
                ):
                    q_values[action] = sum(
                        p * (r + self.gamma * self.V[t + 1][s_next])
                        for p, s_next, r in self.env.transitions(s, action)
                    )

                self.V[t][s] = max(q_values.values())

    def get_value(self, state: int, time_step: int) -> float:
        return self.V[time_step][state]

    def get_action(self, state: int, time_step: int) -> ShuttleEnvironmentAction:
        q_values: dict[ShuttleEnvironmentAction, float] = {
            action: sum(
                p * (r + self.gamma * self.V[time_step + 1][s_next])
                for p, s_next, r in self.env.transitions(state, action)
            )
            for action in (
                ShuttleEnvironmentAction.DISPATCH,
                ShuttleEnvironmentAction.WAIT,
            )
        }

        return max(q_values, key=q_values.get)
