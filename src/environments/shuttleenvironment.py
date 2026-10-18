from enum import Enum
from math import isclose

from .mdpenvironment import MDPEnvironment


class ShuttleEnvironmentAction(Enum):
    DISPATCH = 0
    WAIT = 1


class ShuttleEnvironment(MDPEnvironment[int, ShuttleEnvironmentAction]):
    action = ShuttleEnvironmentAction

    def __init__(
        self,
        shuttle_capacity: int,
        shuttle_dispatch_cost: int,
        customer_waiting_cost: int,
        customer_arriving_options: list[int],
        customer_arriving_probabilities: list[float],
    ) -> None:
        if len(customer_arriving_options) != len(customer_arriving_probabilities):
            raise ValueError(
                "Arrival options and probabilities must have equal lengths."
            )

        if not isclose(sum(customer_arriving_probabilities), 1.0):
            raise ValueError("Arrival probabilities must sum to 1.")

        if any(p < 0 for p in customer_arriving_probabilities):
            raise ValueError("Probabilities cannot be negative.")

        self.shuttle_capacity = shuttle_capacity
        self.shuttle_dispatch_cost = shuttle_dispatch_cost
        self.customer_waiting_cost = customer_waiting_cost
        self.customer_arriving_options = customer_arriving_options
        self.customer_arriving_probabilities = customer_arriving_probabilities

    def transitions(
        self, state: int, action: ShuttleEnvironmentAction
    ) -> list[tuple[float, int, float]]:
        """Return (probability, next_state, reward) outcomes."""

        remaining = (
            max(0, state - self.shuttle_capacity)
            if action == ShuttleEnvironmentAction.DISPATCH
            else state
        )

        dispatch_cost = (
            self.shuttle_dispatch_cost
            if action == ShuttleEnvironmentAction.DISPATCH
            else 0
        )

        # I first dispatch the shuttle (if chosen) and calculate the waiting cost
        # based on the remaining queue. New arrivals are then added to determine
        # the next state; they do not incur waiting costs until the next period.

        return [
            (
                prob,
                remaining + arrivals,
                -(remaining * self.customer_waiting_cost + dispatch_cost),
            )
            for arrivals, prob in zip(
                self.customer_arriving_options,
                self.customer_arriving_probabilities,
            )
        ]
