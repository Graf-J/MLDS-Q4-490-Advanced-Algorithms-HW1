from ..environments.shuttleenvironment import (
    ShuttleEnvironment,
    ShuttleEnvironmentAction,
)
from .mdpsolver import MDPSolver


class ShuttlePolicyIteration(MDPSolver[int, ShuttleEnvironmentAction]):
    def __init__(
        self,
        env: ShuttleEnvironment,
        max_state: int,
        gamma: float,
        theta: float = 1e-6,
        initial_policy: list[ShuttleEnvironmentAction] | None = None,
        max_iterations: int = 100,
    ) -> None:
        super().__init__(env)

        # Policy evaluation only converges for a discount factor strictly below 1
        if not 0 <= gamma < 1:
            raise ValueError("Discount factor gamma must be in [0, 1).")

        if initial_policy is not None and len(initial_policy) != max_state + 1:
            raise ValueError("Initial policy must define an action for every state.")

        self.max_state = max_state
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.initial_policy = initial_policy or [ShuttleEnvironmentAction.WAIT] * (
            max_state + 1
        )

        # Evaluation stops at delta < theta, so V is accurate up to this bound;
        # smaller Q-value differences are treated as ties
        self.improvement_tolerance = theta * gamma / (1 - gamma)

        self.policy: list[ShuttleEnvironmentAction] = list(self.initial_policy)
        self.V: list[float] = [0.0] * (max_state + 1)

        # One entry per iteration: the evaluated policy, its values and the sweeps needed
        self.policy_history: list[list[ShuttleEnvironmentAction]] = []
        self.value_history: list[list[float]] = []
        self.evaluation_sweeps: list[int] = []

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

    def evaluate_policy(self) -> int:
        """Iterative policy evaluation of the current policy; returns the number of sweeps."""
        # Warm start from the values of the previous policy
        sweeps = 0

        while True:
            delta = 0.0
            V_next = self.V.copy()

            for s in range(self.max_state + 1):
                v = sum(
                    p * (r + self.gamma * self.V[min(s_next, self.max_state)])
                    for p, s_next, r in self.env.transitions(s, self.policy[s])
                )
                V_next[s] = v
                delta = max(delta, abs(v - self.V[s]))

            self.V = V_next
            sweeps += 1

            if delta < self.theta:
                return sweeps

    def improve_policy(self) -> bool:
        """Greedy policy improvement; returns True if the policy did not change."""
        policy_stable = True

        for s in range(self.max_state + 1):
            old_action = self.policy[s]
            q_values = self.q_values(s)
            best_action = max(q_values, key=q_values.get)

            # Only switch for a real improvement, so ties cannot make the policy oscillate
            if q_values[best_action] > q_values[old_action] + self.improvement_tolerance:
                self.policy[s] = best_action
                policy_stable = False

        return policy_stable

    def solve(self) -> None:
        self.policy = list(self.initial_policy)
        self.V = [0.0] * (self.max_state + 1)
        self.policy_history, self.value_history, self.evaluation_sweeps = [], [], []

        for _ in range(self.max_iterations):
            self.evaluation_sweeps.append(self.evaluate_policy())
            self.policy_history.append(list(self.policy))
            self.value_history.append(self.V.copy())

            if self.improve_policy():
                return

        raise RuntimeError(
            f"Policy iteration did not converge within {self.max_iterations} iterations."
        )

    def get_value(self, state: int, time_step: int | None = None) -> float:
        return self.V[state]

    def get_action(
        self, state: int, time_step: int | None = None
    ) -> ShuttleEnvironmentAction:
        # The policy is stationary, so the time step is ignored
        return self.policy[state]
