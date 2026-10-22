# Repository and How to Run the Code

- **GitHub username:** Graf-J
- **Repository:** [MLDS-Q4-490-Advanced-Algorithms-HW1](https://github.com/Graf-J/MLDS-Q4-490-Advanced-Algorithms-HW1) (public)

The project uses [uv](https://docs.astral.sh/uv/) for Python (3.14) and its dependencies:

```bash
git clone https://github.com/Graf-J/MLDS-Q4-490-Advanced-Algorithms-HW1.git
cd MLDS-Q4-490-Advanced-Algorithms-HW1
uv sync            # creates .venv with the exact versions from uv.lock
uv run jupyter lab # or open the notebooks in VS Code with the .venv kernel
```

# 2 Value and Policy Iteration

## 2.0 Model and Shared Environment

| Element | Definition |
|---|---|
| State $s_t$ | Number of customers waiting at the start of period $t$ |
| Action $a_t$ | `DISPATCH` or `WAIT` |
| Transition | $s_{t+1} = \max(0,\ s_t - K \cdot \mathbb{1}[\text{dispatch}]) + A_t$, with $A_t \sim \text{Unif}\{1, \dots, 5\}$ |
| Reward | $r_t = -\big(c_f \cdot \mathbb{1}[\text{dispatch}] + c_h \cdot \max(0,\ s_t - K \cdot \mathbb{1}[\text{dispatch}])\big)$ |
| Parameters | $K = 15$, $c_f = 100$, $c_h = 2$ |

**Order of events within a period:** (1) decide whether to dispatch, (2) pay $c_h$ for every customer left behind, (3) new customers arrive and are first charged in the next period.

The environment (`src/environments/shuttleenvironment.py`) is shared by all subparts of Part 2. Its method `transitions(s, a)` returns every possible outcome as a `(probability, next_state, reward)` tuple, and all solvers use only this method.

### Environment Validation (`2_ValidateEnvironment.ipynb`)

Starting from $s_0 = 0$, I propagated the exact state distribution $P(S_t = s)$ for 20 steps, using fixed policies whose outcome can be predicted by hand:

| Arrivals | Always `DISPATCH` | Always `WAIT` |
|---|---|---|
| Always 1 | $s_t = 1$ ✓ | $s_t = t$ ✓ |
| Always 5 | $s_t = 5$ ✓ | $s_t = 5t$ ✓ |
| Unif$\{1..5\}$ | Uniform on $\{1..5\}$, mean 3 ✓ | Support $[t, 5t]$, mean $3t$ ✓ |

In every case, the probabilities sum to 1 at each step.

![Environment validation: always WAIT with uniform arrivals](figures/env_wait_uniform.png)
![Environment validation: always DISPATCH with uniform arrivals](figures/env_dispatch_uniform.png)

---

## 2a) Enumeration ($T = 500$)

**Notebook:** `2a_Enumeration.ipynb`. **Solver:** `src/solvers/shuttleenumeration.py`.

**Method:** backward induction. Starting from $V_T \equiv 0$, I compute for $t = T-1, \dots, 0$

$$V_t(s) = \max_{a} \mathbb{E}_{A_t}\big[r(s, a) + V_{t+1}(s')\big],$$

and the optimal action is the corresponding $\arg\max$. The solve takes about 6 s.

**Assumptions:**
- Initial states $s_0 \in \{0, \dots, 50\}$.
- No discounting ($\gamma = 1$): the horizon is finite, so all values are finite anyway. $\gamma = 0.95$ is used in 2b and 2c.
- No terminal cost ($V_T \equiv 0$).
- No cap at 200 customers: every reachable state is kept (see the edge cases).

### Optimal Value Function at $t = 0$

![Optimal value function at t = 0](figures/2a_value_function.png)

- **Decreasing in $s$:** more waiting customers always means more future cost. $V_0(0) = -16{,}071.6$, about 32 per period.
- **Flat for $s = 13$–$15$:** here $V_0(s) = V_0(0) - 100$ exactly. The optimal action dispatches, the shuttle takes everyone, and the system is back at an empty queue for the price of one dispatch.
- **Kinks at multiples of $K = 15$:** each further block of 15 customers needs one more shuttle.

### Optimal Policy and Simulated Behaviour

![Optimal policy over time and queue size](figures/2a_policy_map.png)

The optimal policy is a **threshold policy: dispatch if and only if $s \geq 13$**, for $t = 0, \dots, 489$. Near the end of the horizon:
- The threshold briefly drops to 12.
- In the last 3 periods, the policy never dispatches. A dispatch removes at most 15 customers and therefore saves at most $3 \cdot 15 \cdot c_h = 90 < c_f = 100$.

![Simulated trajectories under the optimal policy](figures/2a_trajectories.png)

Simulating the policy from $s_0 \in \{0, 10, 25, 50\}$ shows exactly the expected behaviour:
- **Large initial queues are brought down immediately.** From $s_0 = 50$, the policy dispatches in four consecutive periods ($50 \to 39 \to 29 \to 16 \to 4$).
- **Afterwards, all trajectories follow the same sawtooth.** The queue builds up until it reaches 13 and is then emptied.
- **The remaining variation is only due to random arrivals.** Because at most 5 customers arrive per period, the queue at a dispatch is always between 13 and $12 + 5 = 17$ (observed: 13–17, mean 14.3). The time between dispatches ranges from 3 to 8 periods, with a mean of 4.7, which matches $\approx 14.3 / \mathbb{E}[A] = 14.3 / 3$.
- **At the end of the horizon, queues grow unchecked** (16–23 customers at $T$), consistent with the policy map.

### Edge Cases

| Edge case | Handling |
|---|---|
| Fewer than $K$ customers (or none) when dispatching | The remaining queue is $\max(0, s - K)$, so it is never negative. The dispatch still costs $c_f$, so the optimal policy never dispatches an empty station unless $c_f = 0$. |
| Unbounded queue growth | At most 5 customers arrive per period, so `V[t]` holds the $50 + 5t + 1$ reachable states (the $+1$ because `range` excludes the upper bound). Every successor state therefore exists: exact values, no truncation or boundary effects. The 200 cap is never needed, since the optimal policy keeps the queue at 17 or below. |
| Invalid arrival distribution | A `ValueError` is raised for mismatched lengths, negative probabilities, or probabilities that do not sum to 1. The sum is checked with `math.isclose`, so rounding errors such as $5 \times 0.2$ are accepted. |
| Ties between actions | `get_action` returns the first maximiser (`DISPATCH`). Ties only occur in degenerate cases, e.g. $c_f = 0$ at $s = 0$. |
| End of the horizon | $V_T \equiv 0$ makes leftover customers free, so the policy stops dispatching in the last 3 periods. This is a property of the model, not a bug. |

### Tests for Validating Correctness

The tests are in the **Correctness Checks** section of `2a_Enumeration.ipynb` and run top to bottom with the notebook. Each test prints `[PASS]` / `[FAIL]`, and all of them pass.

**1. Remove the stochasticity.** With exactly 3 arrivals per period, the problem is deterministic, so the optimum is the best of all $2^T$ action sequences. For $T = 12$ (4,096 sequences) and every $s_0 \in \{0, \dots, 30\}$, the DP value equals the brute-force optimum exactly (max error 0).

![Deterministic arrivals: DP vs. brute force](figures/2a_brute_force.png)

**2. Extreme parameters with obvious answers** ($T = 60$):

| Setting | Expected and observed |
|---|---|
| $c_h = 0$ | Never dispatch, $V \equiv 0$ |
| $c_f = 0$, or $c_h = 200 > c_f$ | Dispatch whenever $s \geq 1$ |
| $K = 0$ | Never dispatch, $V_0(0) = -c_h\,\mathbb{E}[A]\,\frac{T(T-1)}{2} = -10{,}620$ |

![Optimal policy in extreme cases](figures/2a_extreme_cases.png)

**3. Parameter sweeps.** I changed one parameter at a time and recorded the dispatch threshold at $t = 0$:

![Dispatch threshold under parameter sweeps](figures/2a_parameter_sweeps.png)

| Change | Threshold | Reason |
|---|---|---|
| Dispatch cost $c_f$ ↑ | Rises, 0 → 15 | An expensive shuttle is worth filling up first. |
| Waiting cost $c_h$ ↑ | Falls, 15 → 1 | Expensive waiting forces earlier dispatches. |
| Capacity $K$ ↑ | Rises, then flat at 16 | Once the shuttle is large, capacity stops being the bottleneck. |

The threshold never exceeds $K$, and the policy stays threshold-shaped in all 30 variants.

**4. First 10 steps.** For one simulated trajectory, I checked at every step that $V_t(s_t) = \max_a Q_t(s_t, a)$, that the action is the $\arg\max$, and that the reward matches the formula.

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| $s_t$ | 0 | 3 | 8 | 9 | **14** | 2 | 5 | 10 | **13** | 3 |
| $Q(\text{D}) - Q(\text{W})$ | −100.0 | −66.9 | −25.0 | −19.1 | **+17.7** | −77.2 | −48.5 | −14.3 | **+7.2** | −66.9 |
| Action | W | W | W | W | **D** | W | W | W | **D** | W |
| Reward | 0 | −6 | −16 | −18 | −100 | −4 | −10 | −20 | −100 | −6 |

![First 10 steps: state, dispatch advantage and reward](figures/2a_first_10_steps.png)

The advantage of dispatching grows as the queue fills up. It turns positive exactly when the threshold of 13 is reached, and the queue is then emptied.

**5. Optimal vs. other policies.** Starting from $s_0 = 0$, I compared $V^*$ with simulated rollouts and with the *exact* expected values (computed by exact policy evaluation) of a random policy and of every fixed threshold rule "dispatch if $s \geq k$":

![Optimal policy vs. random and fixed-threshold policies](figures/2a_optimal_vs_random.png)

| Policy | Expected total reward |
|---|---|
| **Optimal, $V^*_0(0)$** | **−16,071.6** |
| Optimal, mean of 50 rollouts | −16,120.5 (sampling noise around $V^*$) |
| Best fixed threshold, $k = 13$ | −16,099.9 |
| Random, dispatching with probability 0.5 | −28,299.9 (all 50 rollouts below $V^*$) |

No other policy beats $V^*$. The best fixed threshold is the one the DP found ($k = 13$); it is slightly worse than $V^*$ only because the optimal policy also adapts near the end of the horizon.

---

## 2b) Value Iteration ($T = \infty$)

**Notebook:** `2b_ValueIteration.ipynb`. **Solver:** `src/solvers/shuttlevalueiteration.py`.

**Method:** starting from $V \equiv 0$, I repeatedly apply the Bellman optimality update to every state $s \in \{0, \dots, 200\}$:

$$V(s) \leftarrow \max_{a} \mathbb{E}_{A}\big[r(s, a) + \gamma\,V(s')\big].$$

The values are updated in place, so later states within a sweep already use the new values. I stop when the largest change in a sweep, $\Delta_k = \max_s |V_k(s) - V_{k-1}(s)|$, drops below $10^{-8}$; this takes 343 sweeps. The policy is the greedy $\arg\max$ with respect to the final $V$.

**Assumptions:**
- Discount factor $\gamma = 0.95$.
- At most 200 customers: a next state above 200 is clamped to 200, i.e. extra customers are turned away (see the edge cases).

### Optimal Value Function

![Optimal value function from value iteration](figures/2b_value_function.png)

- **Decreasing in $s$:** as in 2a, more waiting customers always means more future cost. $V^*(0) = -576.7$, i.e. $(1 - \gamma)\,V^*(0) \approx -28.8$ per period. This is lower than 2a's 32 per period because discounting weights the cheap first periods after starting from an empty station most.
- **Same shape as 2a:** flat at $s = 13$–$15$, with kinks at multiples of $K = 15$. Tests 1–3 explain these features exactly.

### Optimal Policy and Simulated Behaviour

The optimal policy is again the **threshold policy: dispatch if and only if $s \geq 13$**. This is the same threshold as in 2a, and it holds for all 201 states.

![Simulated trajectories under the value-iteration policy](figures/2b_trajectories.png)

The simulation from 2a, repeated with the value-iteration policy, shows the same behaviour:
- Large queues are cleared immediately, then every trajectory follows the 13 → dispatch sawtooth.
- Every dispatch happens at a queue of 13–17 (mean 14.3), with 3–8 periods between dispatches (mean 4.7).
- **Unlike 2a, there is no end-of-horizon effect.** The policy does not depend on time, so the queue stays at 17 or below until $t = 500$ and beyond. This confirms that the behaviour at the end of 2a came from the finite horizon.

### Edge Cases

The edge cases of the shared environment (dispatching fewer than $K$ customers, invalid arrival distributions) are handled as described in 2a.

| Edge case | Handling |
|---|---|
| Next state above the cap of 200 | `V[min(s', 200)]`: the queue cannot exceed 200, and extra customers are turned away. Doubling the cap to 400 leaves $V^*$ on $s \leq 50$ unchanged (difference exactly 0), so the cap does not affect the states that occur. |
| Invalid discount factor | Value iteration only converges for $0 \leq \gamma < 1$ (with $\gamma = 1$ the infinite sum of costs diverges). Any other value raises a `ValueError`; tested for $\gamma = 1$, $1.5$ and $-0.1$. |
| When to stop | There is no fixed number of sweeps. The solver stops at $\Delta_k < 10^{-8}$. |
| Initial values | $V \equiv 0$. Any starting point converges to the same $V^*$; starting from 0 makes sweep $k$ comparable to a $k$-period enumeration (Test 6). |
| Ties between actions | As in 2a: `get_action` returns `DISPATCH` on a tie, which only happens in degenerate cases. |

### Tests for Validating Correctness

The tests are in the **Checks** section of `2b_ValueIteration.ipynb` and run top to bottom with the notebook. Each test prints `[PASS]` / `[FAIL]`, and all of them pass. Tests 1–3 check that the structure of $V^*$ matches what the model predicts, Tests 4–5 check the iteration itself, and Tests 6–7 compare against two independent methods.

**1. What does one more customer cost?** $V^*(s) - V^*(s-1)$ is the value of one extra waiting customer.

![Cost of one more waiting customer](figures/2b_marginal_cost.png)

- It is never positive, and it shrinks as the queue approaches the threshold, because a shuttle is coming soon.
- It is **exactly 0 for $s = 14, 15$:** at $s \geq 13$ the policy dispatches anyway, and up to $K = 15$ everyone fits in, so an extra customer just rides along.
- At $s = 16$ it jumps back to $-10.77$, exactly the cost of the very first customer: the 16th customer is left behind alone.

**2. Where does DISPATCH take over from WAIT?** For every $s \leq K$, a dispatch takes everyone, so the system afterwards is exactly where an empty station would be after waiting:

$$Q(s, \text{DISPATCH}) = -c_f + \gamma\,\mathbb{E}[V^*(A)] = V^*(0) - c_f = -676.67 \quad \text{for all } s \leq K.$$

![Q-values and the switch from WAIT to DISPATCH](figures/2b_q_values.png)

$Q(\text{WAIT})$ decreases with every customer and drops below this constant floor at $s = 13$, the threshold. This also matches 2a.

**3. The value function repeats every $K = 15$ customers.** Dispatching turns $s$ customers into $s - K$ for a price of $c_f$. If the $s - K$ customers left behind are below the threshold, then $V^*(s) = V^*(s - K) - c_f$ exactly.

![The value function repeats every K customers](figures/2b_periodicity.png)

- This holds exactly for $s = 15$–$27$.
- From $s = 28$ on, the customers left behind are themselves at the threshold and need their own shuttle soon, so the gap grows beyond 100.

**4. Convergence.** The Bellman update is a $\gamma$-contraction, so $\Delta_k$ must decay at least as fast as $\gamma^k$, which is a straight line on a log scale.

![Convergence of value iteration](figures/2b_convergence.png)

- After a short initial phase, $\Delta_k$ shrinks by a factor of 0.937 per sweep (median over the last 100 sweeps).
- This is slightly faster than $\gamma = 0.95$ because of the in-place updates.

**5. How the Q-values and the policy evolve.** After every sweep $k$, the current $V_k$ defines a greedy policy. The heatmap shows $Q_k(\text{DISPATCH}) - Q_k(\text{WAIT})$ for every sweep and state:

![Evolution of the dispatch advantage over the sweeps](figures/2b_q_heatmap.png)

- **Sweeps 1–2:** $V \approx 0$ ignores the future, and waiting (at most $2 \cdot 15 = 30$ per period) looks cheaper than a dispatch (100), so the policy never dispatches.
- **Sweeps 3–34:** the threshold jumps to 14, overshoots down to 10, and then climbs back via 11 and 12.
- **From sweep 35 of 343:** the policy is final at 13, although the values are still up to 77 away from $V^*$. Only the sign of the advantage matters for the action, which is exactly what policy iteration (2c) exploits.

**6. Cross-check with enumeration (2a).** I ran the enumeration solver from 2a with the same $\gamma = 0.95$. Its value for horizon $n$, $V_0^{(n)}$, should approach $V^*$ like $\gamma^n$, since backward induction from $V \equiv 0$ is value iteration started from 0.

![Discounted enumeration converges to V*](figures/2b_enumeration_crosscheck.png)

- The gap stays below $\gamma^n \cdot$ (initial gap) for every horizon $n$, reaching $1.4 \cdot 10^{-4}$ at $n = 300$.
- Both solvers choose the same action in every state $s \leq 50$.

---

## 2c) Policy Iteration

**Notebook:** `2c_PolicyIteration.ipynb`. **Solver:** `src/solvers/shuttlepolicyiteration.py`.

**Method:** starting from the policy $\pi_0$ = "always wait", I alternate two steps until the policy no longer changes:
1. **Policy evaluation:** repeat $V(s) \leftarrow \mathbb{E}_{A}\big[r(s, \pi(s)) + \gamma\,V(s')\big]$ for all states until the largest change is below $\theta = 10^{-6}$. Each evaluation starts from the values of the previous policy (warm start).
2. **Policy improvement:** set $\pi(s) \leftarrow \arg\max_a \mathbb{E}_A\big[r(s, a) + \gamma\,V(s')\big]$. An action is only switched if the new one is better by more than the evaluation error $\gamma\theta/(1-\gamma) = 1.9 \cdot 10^{-5}$.

This converges after **6 policy evaluations**.

**Assumptions:** as in 2b ($\gamma = 0.95$, next states clamped at 200), plus the evaluation tolerance $\theta = 10^{-6}$ and the initial policy "always wait".

### Optimal Policy

![Optimal policy from policy iteration](figures/2c_policy.png)

The optimal policy is the **threshold policy: dispatch if and only if $s \geq 13$**. It is identical to value iteration in all 201 states and to the steady-state policy of enumeration. The value function also matches 2b (max difference $1.9 \cdot 10^{-5}$), so it is not shown again.

### Edge Cases

The edge cases of the environment (2a) and of the state cap (2b) apply unchanged; doubling the cap to 400 again leaves policy and values on $s \leq 50$ unchanged.

| Edge case | Handling |
|---|---|
| Ties and evaluation noise | Policy evaluation is only accurate up to $\gamma\theta/(1-\gamma)$. If the solver switched to any action that looks marginally better, two (nearly) equally good actions could swap back and forth forever. It therefore switches only for an improvement larger than this error. Test: with $c_f = c_h = 0$, every Q-value is 0, and the solver keeps its initial policy and stops after one evaluation. |
| Termination | Policy iteration terminates in theory, since there are finitely many policies and each iteration strictly improves. As a safeguard, `max_iterations` (default 100) raises an error instead of looping forever; tested with a limit of 2. |
| Inaccurate policy evaluation | Evaluation stops at $\Delta < \theta$. I checked the result against an exact linear solve (Test 3), and every check that compares values allows for this error. |
| Initial policy | Any policy can be passed in. One that does not cover every state raises a `ValueError`. The result does not depend on it (Test 4). |
| Initial policy "always wait" | This policy never empties the station, so its value depends on the cap: at $s = 200$ it pays $c_h \cdot 200$ forever, $V^{\pi_0}(200) = -2 \cdot 200 / (1 - \gamma) = -8{,}000$ exactly. The clamp keeps this finite and well-defined. |
| Invalid discount factor | As in 2b, $\gamma \notin [0, 1)$ raises a `ValueError`. |

### Tests for Validating Correctness

The tests are in the **Checks** section of `2c_PolicyIteration.ipynb` and run top to bottom with the notebook. Each test prints `[PASS]` / `[FAIL]`, and all of them pass.

**1. How the policy evolves.** Each row shows the policy evaluated in one iteration:

![Policy after each iteration](figures/2c_policy_evolution.png)

| Iteration | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| First state with DISPATCH | never | 3 | 26 | 7 | 14 | **13** |
| $V^{\pi_k}(0)$ | −2,204 | −1,330 | −891 | −749 | −582 | **−577** |

- **$\pi_1$ dispatches almost always.** Under "always wait", the queue grows forever, so $V^{\pi_0}$ is very low and dispatching looks attractive almost everywhere.
- **Intermediate policies need not be thresholds.** $\pi_2$ dispatches at 26–29, waits at 30–32, and dispatches again from 33, because it acts greedily on the values of a bad policy.
- **The policy settles in a few steps.** It narrows down to 14 and then to 13. Every iteration except the last changes the policy.

**2. Policy improvement theorem.** Acting greedily on $V^{\pi_k}$ can never make things worse, so $V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s)$ must hold for every state and iteration.

![Value of each policy](figures/2c_policy_improvement.png)

- The smallest improvement over all states per iteration is $+160$, $+110$, $+79$ and $+2.5$.
- The first step shows $-1.8 \cdot 10^{-5}$ at $s = 199$, where both policies wait. This is within the evaluation error.
- The value curves end on $V^*$ from value iteration.

**3. Agreement with exact evaluation and value iteration.** For a fixed policy, the Bellman equation is linear, so $V^\pi$ can also be computed exactly by solving $(I - \gamma P_\pi)\,V = r_\pi$ (a $201 \times 201$ system).

![Agreement with exact evaluation and value iteration, and computational effort](figures/2c_agreement_and_effort.png)

- Policy iteration chooses the same action as value iteration in all 201 states.
- Its values differ from the exact solution by at most $1.86 \cdot 10^{-5}$ and from value iteration by at most $1.87 \cdot 10^{-5}$. Both are below the evaluation error bound of $1.9 \cdot 10^{-5}$.
- **Effort:** policy iteration needs only 6 policies, but evaluating them iteratively costs 1,930 sweeps in total, compared with 343 sweeps for value iteration. Each iterative evaluation is itself a value-iteration-like process to tolerance $\theta$, and the values change a lot between policies, so the warm start helps little. The small number of improvement steps would only pay off with cheaper evaluation, e.g. one exact linear solve per policy, or a looser $\theta$ in early iterations.

**4. Independence of the initial policy.** The optimal policy is unique, so every starting point must lead to it:

![Policy iteration from different initial policies](figures/2c_initial_policies.png)

| Initial policy | Evaluations | Evaluation sweeps | First state with DISPATCH per iteration |
|---|---|---|---|
| Always wait (default) | 6 | 1,930 | never → 3 → 26 → 7 → 14 → 13 |
| Always dispatch | 5 | 1,600 | 0 → 45 → 5 → 14 → 13 |
| Dispatch if $s \geq 1$ | 5 | 1,600 | 1 → 45 → 5 → 14 → 13 |
| Dispatch if $s \geq 30$ | 4 | 1,231 | 30 → 7 → 14 → 13 |
| Random (each state 50/50) | 4 | 1,236 | 1 → 17 → 11 → 13 |

All five runs end in exactly the policy from value iteration after at most 6 evaluations.

### Summary of Part 2

| | Enumeration (2a) | Value iteration (2b) | Policy iteration (2c) |
|---|---|---|---|
| Horizon, discount | $T = 500$, $\gamma = 1$ | $\infty$, $\gamma = 0.95$ | $\infty$, $\gamma = 0.95$ |
| Optimal policy | Dispatch if $s \geq 13$ (except the last ~10 periods) | Dispatch if $s \geq 13$ | Dispatch if $s \geq 13$ |
| Work until done | 1 backward pass, 500 steps | 343 sweeps (policy final after 35) | 6 policies, 1,930 evaluation sweeps |

All three methods agree on the same threshold policy. The tests above confirm it in different ways: brute force, closed-form properties of $V^*$, convergence theory, and exact policy evaluation.

---

## 2.4 Question: "Given an Additional 3 Months, What Other Validation Tests and Experiments Would You Perform?"

The problem is small (201 states, 2 actions), and three independent methods already agree, so 3 months is more than this instance needs. I would use the time to check whether the results hold **beyond this instance** and whether the **model assumptions** are right:

- **Random problem instances:** run all three solvers on thousands of random parameter sets ($K$, $c_f$, $c_h$, arrival distributions) and automatically check the invariants from this report: the solvers agree, the policy is a threshold policy, $V^*$ decreases in $s$, and brute force matches on small instances.
- **Discount factor towards 1:** $(1-\gamma)\,V^*(0)$ should approach the long-run average cost of about 32 per period found by enumeration in 2a. This would connect the finite-horizon and the discounted results.
- **Cheaper policy evaluation:** replace the iterative evaluation in 2c, which dominated the runtime, with an exact linear solve per policy. Then compare runtimes on larger state spaces.
- **Stress the model:**
  - High load ($\mathbb{E}[A] > K$), where the cap of 200 actually binds.
  - Time-varying (rush-hour) arrivals, where the threshold should change over the day.
  - A policy tuned for the wrong arrival distribution, to see how much reward it loses.
- **Benchmark for learning algorithms:** since the exact optimum is known, check whether model-free methods (Q-learning, policy gradient) learn the threshold of 13 from simulation alone.

# 3 Policy Gradient: Part 1

## 3a) CartPole-v1

**Notebook:** `3a_CartPole-v1.ipynb`. All figures are saved to `report/figures/3a_*.png` when the notebook runs. The four main trainings take about 12 min; the seed experiment takes another 45 min.

**Method:** REINFORCE (vanilla policy gradient) without a baseline.
- **Policy network:** $4 \to 20$ (ReLU) $\to 1$ logit, with $P(\text{push right} \mid s) = \sigma(\text{logit})$. During training, the action is sampled from this Bernoulli distribution.
- **Update:** after every episode, I compute the discounted returns $G_t = \sum_k \gamma^k r_{t+k}$ and take one Adam step (learning rate 0.01) on the loss $-\sum_t \log \pi(a_t \mid s_t)\, G_t$. The returns are not normalised.
- **Training:** 1,000 episodes. All four state variables start uniformly in $[-0.15, 0.15]$; this is wider than the Gymnasium default of $\pm 0.05$, so the policy also learns to recover from harder start states.
- **Evaluation:** 500 episodes with the greedy action (push right iff $P > 0.5$). Episode $i$ starts from seed $i$, so every variant is evaluated on the same 500 start states.

The function `train(env, policy, optimizer, episodes, gamma, scheduler=None, reward_shaping=False, seed=None)` supports two optional extensions. I add them one at a time on top of the baseline, which uses $\gamma = 0.95$ as the assignment requires:

| Variant | $\gamma$ | Learning rate | Reward used for learning |
|---|---|---|---|
| Baseline | 0.95 | 0.01, constant | Environment reward (+1 per step) |
| 1: bigger $\gamma$ | 0.995 | 0.01, constant | Environment reward |
| 2: + scheduler | 0.995 | Cosine annealing 0.01 → 0.001 over the 1,000 episodes | Environment reward |
| 3: + reward shaping | 0.995 | Cosine annealing 0.01 → 0.001 | $r_t \cdot \varphi(x_{t+1})$, with $\varphi$ the density of $\mathcal{N}(0, 0.7^2)$ |

The shaped reward is $0.570$ with the cart at the center and $0.0016$ at the edge of the track ($|x| = 2.4$), so it pays more for staying centered. It is only used to compute the returns. **All reported rewards are the unshaped environment rewards**, so the variants are directly comparable.

### Baseline: Training and Evaluation

![Baseline training curve](figures/3a_training_baseline.png)

- **Slow and unstable learning:** the moving average stays between 100 and 280 for 900 episodes and collapses once around episode 620. It reaches 475 (the threshold at which CartPole-v1 counts as solved) only at episode 991.
- **During training, the cart leaving the track is the main failure:** this happens in 564 of the 1,000 episodes (Test 6).

![Baseline evaluation histogram](figures/3a_histogram_baseline.png)

**Evaluation:** all 500 greedy episodes reach the time limit of 500 steps, so **mean = 500.0, std = 0.0**.

### Variants: Bigger $\gamma$, Scheduler, Reward Shaping

![Training curves of all variants](figures/3a_training_variants.png)
![Moving averages of all variants](figures/3a_moving_averages.png)

| Variant | Solved at episode | Mean reward over training | Mean of last 100 | Evaluation mean ± std |
|---|---|---|---|---|
| Baseline ($\gamma = 0.95$) | 991 | 167.0 | 483.1 | 500.0 ± 0.0 |
| $\gamma = 0.995$ | 669 | 349.2 | 481.4 | 500.0 ± 0.0 |
| + scheduler | 523 | 333.2 | 486.9 | 500.0 ± 0.0 |
| + reward shaping | **341** | **427.9** | **492.9** | 500.0 ± 0.0 |

The evaluation histograms of the variants look the same as the baseline's: all 500 episodes reach 500 (`figures/3a_histogram_*.png`).

- **Bigger $\gamma$:** learning is much faster (more than doubles the mean reward over training), but unstable. The moving average reaches 470 at episode 330 and then collapses to 240.
- **+ Scheduler:** it still dips once while the learning rate is high (to 370 around episode 690). From episode ~780 on, the learning rate is small enough that it stays above 475, whereas $\gamma = 0.995$ alone collapses again at episode 900.
- **+ Reward shaping:** fastest and most stable. From episode 300 on, almost every episode survives 500 steps (81% of all training episodes).

**For seed 0, all four variants end with a perfect greedy policy.** The hyperparameters therefore improve **how fast and how reliably** training gets there, which the following experiment and Tests 5–6 confirm.

### Robustness Across Training Seeds

One training run can be lucky or unlucky, so I trained every variant with 4 more seeds and evaluated each policy greedily on 100 start states:

![Mean training curve over 5 seeds](figures/3a_seeds.png)

| Variant | Solved at (seeds 0–4) | Mean training reward | Mean of last 100 | Evaluation mean | Seeds with evaluation = 500 |
|---|---|---|---|---|---|
| Baseline ($\gamma = 0.95$) | 991, 525, –, –, – | 192.8 ± 60.4 | 292.4 ± 138.0 | 362.0 | 3/5 |
| $\gamma = 0.995$ | 669, 333, 395, –, 662 | 286.3 ± 78.3 | 427.6 ± 127.6 | 462.2 | 4/5 |
| + scheduler | 523, 753, 806, 319, 995 | 315.6 ± 65.3 | 483.6 ± 7.4 | 497.7 | 4/5 |
| + reward shaping | 341, 327, 659, 494, 604 | **363.3 ± 48.3** | **494.6 ± 6.8** | **500.0** | **5/5** |

(– = not solved within 1,000 episodes)

- **Each addition helps on average.** Seed 0 was a lucky run for the baseline: it reaches 475 in only 2 of 5 seeds.
- **The bigger $\gamma$ mostly speeds up learning.** The scheduler mostly makes the end of training reliable: the spread of the last-100 mean drops from ±127.6 to ±7.4.
- **Reward shaping is the only variant that solves the task and reaches a perfect greedy policy for every seed.**

### Edge Cases

| Edge case | Handling |
|---|---|
| Invalid arguments | `episodes <= 0`, $\gamma \notin (0, 1]$ and a scheduler that wraps a different optimizer raise a `ValueError` (tested). $\gamma = 0$ is excluded because every return would then be the immediate reward of 1, which gives no signal about survival. |
| Time limit (truncation at 500 steps) | Truncated episodes are treated like terminated ones: the return is cut off, even though the pole is still upright. With $\gamma = 0.95$, this only affects the last ~60 steps (Test 1). The limit also caps the evaluation: no reward can exceed 500, and std = 0 means "every episode reached the limit", not a bug. Tested: every reward of 500 is a truncation, every shorter episode a failure. |
| Saturated evaluation | Since all variants reach 500 in the greedy evaluation, I compare them under harder conditions (Test 5) and over 5 seeds. |
| Reward shaping must not make failing attractive | The shaped reward is strictly positive ($\geq 0.0016$) on the whole track, so surviving longer always increases the return. With a negative shaped reward (e.g. $1 - |x|$), ending the episode early could become optimal. Tested over the whole track. |
| Shaped vs. true reward | The shaped reward is only used to compute the returns. All plots and tables use the environment reward. |
| Scheduler | Stepped once per episode with `T_max = episodes`, so the learning rate ends exactly at 0.001 (tested). It is only stepped if one is passed. |
| Scale of the returns | Returns reach up to $1/(1-\gamma)$: 20 for $\gamma = 0.95$ but 200 for $\gamma = 0.995$. Adam's update does not change when all gradients are scaled by a constant factor, so the same learning rate works for both without retuning. |
| Start states | $\pm 0.15$ instead of the default $\pm 0.05$, for training and evaluation. A pole angle of 0.15 rad is already close to the failure angle of 0.21 rad. Starting at $\pm 0.2$ is tested in Test 5. |
| Moving average at the start | The first 99 episodes average over fewer episodes. "Solved at" only counts full 100-episode windows. |
| Reproducibility | `torch.manual_seed` and the seed of the first `env.reset` make training reproducible; training twice gives identical rewards (tested). |

### Tests for Validating Correctness

The tests are in the **Checks** section of `3a_CartPole-v1.ipynb` and run top to bottom with the notebook. They reuse the trained policies from above. Each test prints `[PASS]` / `[FAIL]`, and all of them pass.

**1. Discounted returns and the credit horizon.** For a constant reward of 1, $G_t = \frac{1 - \gamma^{T-t}}{1 - \gamma}$ must hold, and with $\gamma = 1$, $G_t$ must equal the number of remaining steps. `rewards_to_returns` matches both exactly.

![Discounting: weight of future rewards and returns of a failing episode](figures/3a_discounting.png)

| $\gamma$ | Effective horizon $1/(1-\gamma)$ | $\gamma^{100}$ |
|---|---|---|
| 0.95 | 20 steps | 0.006 |
| 0.99 | 100 steps | 0.366 |
| 0.995 | 200 steps | 0.606 |

With $\gamma = 0.95$, an episode that fails at step 400 has **the same returns as one that survives all 500 steps for the first ~340 steps**. A slow drift of the cart, which only causes a failure much later, is therefore barely punished. With $\gamma = 0.995$, the later failure lowers the return already at step 0 (173 vs. 183). Test 6 checks whether this explains the cart failures of the baseline.

**2. Control experiments.** Three short trainings (300 episodes, $\gamma = 0.95$) that only differ in the learning signal:

![Control experiments](figures/3a_control_experiments.png)

| Training | First 100 episodes | Last 100 episodes | Expected |
|---|---|---|---|
| Normal | 50.9 | 123.2 | Clearly increases ✓ |
| Learning rate 0 | 20.9 | 21.7 | Flat (the policy never changes) ✓ |
| Negated reward (−1 per step) | 15.4 | 10.2 | Learns to fail as fast as possible ✓ |
| *Reference:* random policy | 19.9 | | |
| *Reference:* always push in one direction | 9.3 | | |

With the reward flipped, the same code learns the opposite behaviour, getting close to the fastest simple way to fail. This shows that the gradient points in the right direction, and that the improvement in normal training comes from the learning signal, not from chance.

**3. First 10 steps.** The trained baseline policy, with sampled actions as in training, starting with the pole leaning left ($\theta = -0.138$):

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| $\theta$ | −0.138 | −0.141 | −0.139 | −0.132 | −0.131 | −0.126 | −0.116 | −0.101 | −0.092 | −0.078 |
| $\dot\theta$ | −0.15 | 0.10 | 0.35 | 0.01 | 0.26 | 0.51 | 0.76 | 0.43 | 0.69 | 0.37 |
| $P(\text{right})$ | 0.03 | 0.05 | 0.14 | 0.03 | 0.08 | 0.24 | **0.59** | 0.16 | 0.48 | 0.12 |
| Action | L | L | R | L | L | L | R | L | R | L |

- **The policy pushes towards the side the pole leans to** (left), in 9 of 10 steps, so the cart moves back under the pole. It only becomes unsure when the pole already swings back quickly ($\dot\theta = 0.76$ at $t = 6$).
- **As a result, $|\theta|$ shrinks** from 0.138 to 0.071.
- The shaped reward stays at 0.568–0.570, the maximum, since the cart stays near the center.

**4. Does the policy react correctly to the pole?** I vary the pole angle $\theta$ and the angular velocity $\dot\theta$ one at a time, with all other state variables at 0:

![Policy response to the pole angle and angular velocity](figures/3a_policy_response.png)

- For all four policies, $P(\text{right})$ increases monotonically in both, is below 0.5 at the negative end and above 0.5 at the positive end (8/8 PASS). This is the physically correct direction: when the pole leans or falls to the right, the cart must move right.
- **The $\gamma = 0.995$ policy is biased:** it pushes right with probability 0.96 even when the pole is perfectly upright. This explains its drift in Test 6.

**5. Harder evaluation.** The greedy evaluation is saturated at 500, so I also evaluate each policy with sampled actions, and with wider start states ($\pm 0.2$):

![Evaluation under harder conditions](figures/3a_robustness.png)

| Variant | Greedy, $\pm 0.15$ | Sampled, $\pm 0.15$ | Greedy, $\pm 0.2$ |
|---|---|---|---|
| Baseline ($\gamma = 0.95$) | 500.0 ± 0.0 | 482.2 ± 75.5 | 500.0 ± 0.0 |
| $\gamma = 0.995$ | 500.0 ± 0.0 | 475.0 ± 104.9 | 486.7 ± 78.2 |
| + scheduler | 500.0 ± 0.0 | 484.6 ± 62.2 | 500.0 ± 0.0 |
| + reward shaping | 500.0 ± 0.0 | **492.5 ± 52.8** | 500.0 ± 0.0 |

The reward-shaping policy is the most robust to the noise of sampled actions. The $\gamma = 0.995$ policy, the one with the bias from Test 4, is the only one that fails from wider start states. Overall, the differences between the final policies of seed 0 are small.

**6. Why do episodes end?** An episode ends when the pole falls ($|\theta| > 12°$), when the cart leaves the track ($|x| > 2.4$), or at the time limit. This checks the explanation from Test 1:

![How training episodes end](figures/3a_end_reasons.png)

| Variant | Training: cart left / pole fell / 500 steps | Sampled evaluation: cart / pole / 500 | Median max $|x|$ (sampled evaluation) |
|---|---|---|---|
| Baseline ($\gamma = 0.95$) | **564** / 333 / 103 | 0 / 33 / 467 | 0.66 |
| $\gamma = 0.995$ | 190 / 276 / 534 | 0 / 27 / 473 | 1.25 |
| + scheduler | 266 / 230 / 504 | 0 / 43 / 457 | 0.68 |
| + reward shaping | **14** / 172 / 814 | 0 / 11 / 489 | **0.26** |

![Cart position during one greedy episode](figures/3a_cart_position.png)

- **During training, the explanation holds.** With $\gamma = 0.95$, the cart leaves the track in 564 episodes, and this is the most common outcome in 6 of the 10 blocks of 100 episodes. The pole is balanced, but the agent does not see that the cart drifts towards the edge. With $\gamma = 0.995$ there are only 190 such failures, and with reward shaping only 14.
- **The final policies no longer leave the track** (0 cart failures in the evaluation). However, only reward shaping keeps the cart centered (median max $|x| = 0.26$). The $\gamma = 0.995$ policy drifts steadily to $x = -1.26$ within 500 steps, because of the bias from Test 4. It would probably leave the track in a longer episode; only the time limit hides this.
- **Conclusion:** a longer horizon makes it possible to learn to avoid the drift, but does not guarantee it. Only reward shaping rewards staying centered explicitly, and only then is the learned policy centered.

### Summary of 3a

| | Baseline ($\gamma = 0.95$) | + $\gamma = 0.995$ | + scheduler | + reward shaping |
|---|---|---|---|---|
| Solved (5 seeds) | 2/5 | 4/5 | 5/5 | 5/5 |
| Mean of last 100 episodes (5 seeds) | 292.4 ± 138.0 | 427.6 ± 127.6 | 483.6 ± 7.4 | 494.6 ± 6.8 |
| Greedy evaluation, seed 0 | 500.0 ± 0.0 | 500.0 ± 0.0 | 500.0 ± 0.0 | 500.0 ± 0.0 |
| Main effect | Short horizon: cart drifts off the track | Faster learning | Stable end of training | Fastest, most stable, cart centered |

Plain REINFORCE with $\gamma = 0.95$ can solve CartPole, but slowly and unreliably. Each added hyperparameter fixes a specific weakness, and reward shaping has the largest effect.

---

## 3b) Pong-v5

**Notebook:** `3b_Pong-v5.ipynb`. All figures are saved to `report/figures/3b_*.png`. Training takes about 100 min; afterwards the policy is saved every 250 episodes to `checkpoints/3b/reinforce/` together with the training log. With `RETRAIN = False` (default), the notebook loads this run instead of training again, so the evaluation and all checks can be rerun in about 30 min.

**Method:** REINFORCE without a baseline, $\gamma = 0.99$.
- **Actions:** only 2 (`RIGHT`) and 3 (`LEFT`), as required. In Pong they move the paddle **up** and **down** (Check 2), so I call them that.
- **Input:** the difference of two consecutive frames, each preprocessed with the given `preprocess` function (80×80, 0/1). A single frame shows where the ball is, but not where it is going. The difference shows moving objects as a +1/−1 pair, which encodes the direction (Check 2).
- **Policy network:** Conv(1→8, 3×3) → ReLU → MaxPool(2) → Conv(8→8, 3×3) → ReLU → MaxPool(2) → Linear(2592→64) → ReLU → Linear(64→1), giving the logit of $P(\text{down})$. Actions are sampled from this Bernoulli distribution.
- **Loss:** $-\sum_t \log \pi(a_t \mid s_t)\, G_t \;-\; 0.01 \sum_t \mathcal{H}[\pi(\cdot \mid s_t)]$, with Adam (learning rate 0.001). The returns are not normalised and no baseline is subtracted.
- **Training:** 2,000 episodes. Gradients are accumulated over 5 episodes per optimizer step and clipped to norm 1.
- **Evaluation:** 500 episodes with actions sampled from the policy, as during training. Episode $i$ starts from seed $i$.

### Deviations from the Plain Algorithm and Why

None of these is a baseline: the learning signal is still the raw discounted return of each action.

| Change | Reason |
|---|---|
| Return restarts after every point | $G_t$ only contains the point of the current rally: $G_t = \gamma^{k} r_{t+k}$, where $t + k$ is the next step with a reward. With the plain discounted sum, an action would also be credited with later rallies it has no influence on. |
| Learning rate 0.001 instead of 0.01 | See the development runs below. |
| 5 episodes per update | Averages the very noisy gradient of single episodes. |
| Entropy bonus (0.01) | Keeps the policy from saturating at one action, after which the gradient vanishes. |
| Gradient clipping (norm 1) | The norm before clipping is between ~3 and ~800, so clipping is active in every update (see the diagnostics). It keeps the direction and rescales every update to norm 1, so every batch counts equally. Adam is insensitive to the overall scale of the gradients anyway, so the growing norm does not change the step size. |
| `Bernoulli(logits=...)` instead of `probs=sigmoid(...)` | Exact log-probabilities when the policy saturates (edge cases). |

**Development runs** (short runs before the final one; not part of the notebook):

| Setting | Episodes | Result |
|---|---|---|
| Single frame as input, lr 0.01, update every episode | 500 | Reward stays at about −21 |
| Frame difference, lr 0.001, update every episode | 500 | Reward stays at about −21 |
| + lr 0.0001, 10 episodes per update, entropy bonus, clipping | 500 | $P(\text{down})$ stays at 0.49–0.51: the policy barely changes |
| lr 0.001, 5 episodes per update (final settings) | 300 | Std of $P(\text{down})$ within an episode rises from 0.001 to 0.013: the policy starts to react to the screen |

### Training

![Training curve](figures/3b_training.png)

- **Learning starts late but then continues.** The moving average stays at about −20.5 for the first ~700 episodes and then rises steadily to **−18.1 over the last 100 episodes** (first 100: −20.6).
- **It has not converged.** The largest gain comes in the last 500 episodes, so the 2,000-episode budget, not convergence, limits the result.

![Training diagnostics](figures/3b_training_diagnostics.png)

The diagnostics show *how* learning happens, before it is visible in the reward:
- **Std of $P(\text{down})$ within an episode:** below 0.01 for the first ~700 episodes, i.e. the policy outputs the same probability whatever the screen shows. It then rises to 0.23. The policy only starts to depend on the state from episode ~750, which is exactly when reward and episode length start to rise.
- **Mean $P(\text{down})$** stays between 0.4 and 0.6, so the policy never collapses to a single action. The initial bias of the random network (0.58) is removed within ~100 episodes.
- **Episode length** doubles from ~1,000 to ~2,100 steps: the agent returns the ball more often, so rallies last longer.
- **Gradient norm** grows from ~35 to ~235 (median of the first and last 50 updates). The loss is a sum over all steps of 5 episodes, so longer episodes add more terms, which explains at most a factor of ~2. The rest is probably because the policy now reacts consistently: early on, the steps push in random directions and mostly cancel; later, many steps push the same weights in the same direction. This does not affect training, since clipping and Adam make the step size independent of the gradient norm.

### Evaluation

![Evaluation histogram](figures/3b_histogram.png)

**Mean = −17.66, std = 1.86** over 500 episodes. 95% of the episodes score better than the random policy (−20.4), and the best episode reaches −10. The policy wins about 3 points per game, but still loses every game.

### Edge Cases

| Edge case | Handling |
|---|---|
| Sparse rewards | Only every ~95 steps (median rally of the trained policy) is a reward ≠ 0. The return restart keeps every return in $[-1, 1]$ and assigns each point only to its rally (Check 1). |
| No motion in a single frame | The frame difference shows direction and speed. In the first step of an episode it is all 0 (tested). |
| Restricted action space | Only actions 2 and 3; there is no "stay" action, so the paddle is always moving. |
| `preprocess` modifies its input | The given function writes into a view of the observation. I copy the frame first, so the observation of the environment stays unchanged (tested). |
| Saturated policy | `Bernoulli(probs=sigmoid(50))` gives $\log P(\text{up}) = -15.9$ because PyTorch clamps the probability; the exact value is −50. `logits=` is exact (tested). The entropy bonus keeps the policy away from saturation. |
| End of an episode | Pong-v5 has no relevant step limit (108,000 frames). Every one of 800 evaluation episodes ends by termination when one side has 21 points (tested). |
| Sticky actions | Pong-v5 repeats the previous action with probability 0.25, so games are random even for a fixed policy. Seeding the first `env.reset` and `torch.manual_seed` makes training reproducible: training twice gives identical results (tested). |
| Invalid arguments | `episodes <= 0`, $\gamma \notin (0, 1]$ and `batch_episodes <= 0` raise a `ValueError` (tested). |
| Long training | Checkpoints and log are saved every 250 episodes, so a crash loses at most 250 episodes. |

### Tests for Validating Correctness

The tests are in the **Checks** section of `3b_Pong-v5.ipynb` and reuse the trained policy and its checkpoints. Each test prints `[PASS]` / `[FAIL]`. **All pass except two**, which test a behaviour the policy has not learned (Checks 4 and 5, explained there).

**1. Returns and credit assignment.** A single final reward gives $G_t = \gamma^{T-1-t}$, and with two rallies `[0, 0, −1, 0, 0, +1]`, the first rally only sees its own point: $[-\gamma^2, -\gamma, -1, \gamma^2, \gamma, 1]$ ✓. In a real 2,837-step episode, $|G_t| \le 1$ at every step ✓.

![Returns in an evaluation episode](figures/3b_returns.png)

With a median rally of 95 steps, $\gamma^{95} = 0.38$: the first action of a rally still gets 38% of the credit for its point.

**2. Preprocessing, frame difference and actions.** The preprocessed frame only contains 0/1 and the difference only −1/0/+1 ✓. Action 2 moves the paddle to the top ($y = 2$ of 158), action 3 to the bottom ($y = 156$) ✓.

![Preprocessing and frame difference](figures/3b_preprocessing.png)

In the difference, the background disappears, and the ball and both paddles appear as +1/−1 pairs. The player's paddle is only visible while it moves.

**3. Comparison with reference policies** (50 episodes each, same start states):

![Reference policies vs. the trained policy](figures/3b_reference_policies.png)

| Policy | Reward | Length (steps) | Points won |
|---|---|---|---|
| Always up / always down | −21.00 ± 0.00 | 764 | 0 |
| Random | −20.40 ± 0.75 | 1,024 | 0.6 |
| Ball tracker (hand-coded) | −14.74 ± 2.16 | 3,876 | 6.3 |
| **Trained, sampled** (first 50 evaluation episodes) | **−18.02 ± 1.83** | 2,217 | 3.0 |
| Trained, greedy | −15.68 ± 2.37 | 2,960 | 5.3 |

- The *ball tracker* is not trained. It always moves the paddle towards the ball, using the ball and paddle positions read from the pixels. It shows what the two actions can achieve and serves as a reference.
- **The trained policy is clearly better than random** ✓ and closes a bit less than half of the gap between random and the ball tracker.
- **Even the ball tracker loses**, because returning the ball is not enough: a point is only won when the opponent misses, which requires steep shots off the edge of the paddle.
- **The greedy policy is better than the sampled one** (−15.68 vs. −18.02). I chose sampled actions for the main evaluation before training, because a policy that is still close to 0.5 everywhere can turn into "always the same action" when acted on greedily. After training, the policy is decisive enough that removing the sampling noise helps.

**4. Does the policy react to the ball?** For every checkpoint (8 episodes each), I compare $P(\text{down})$ in two pairs of situations:
- **Direction gap:** ball moving down vs. ball moving up. The frame difference shows this directly.
- **Position gap:** ball below vs. above the paddle (by more than 10 pixels), only while the ball approaches the player.

Both gaps are 0 for a policy that ignores the ball. I use gaps instead of correlations, because while $P(\text{down})$ barely varies, a correlation only measures noise.

![Policy response to the ball](figures/3b_policy_response.png)

| Checkpoint | 0 | 500 | 750 | 1,000 | 1,500 | 2,000 |
|---|---|---|---|---|---|---|
| Direction gap | 0.000 | +0.003 | +0.026 | +0.095 | +0.089 | **+0.085** |
| Position gap | 0.000 | +0.001 | +0.015 | −0.027 | −0.073 | **−0.057** |

- **Initial policy ignores the ball** (both gaps 0) ✓.
- **The policy learns to move with the ball's vertical direction** ✓. The direction gap appears at episode 750, exactly when the std of $P(\text{down})$ and the reward start to rise.
- **[FAIL] It does not learn to move towards the ball's position.** The position gap never becomes clearly positive and is even negative at the end. This is a property of the learned policy, not a bug in the check: the check is 0 for the untrained policy and works with the same position data that make the ball tracker succeed. The likely reason is the input: the ball's direction is directly visible in the frame difference, but the paddle only appears in it while it moves, so "ball above or below the paddle" is much harder to learn. Following the ball's direction keeps the paddle roughly in line with the ball, which explains the longer rallies. It also explains why the policy stays far below the ball tracker.

**5. First 10 steps.** The 10 steps before the first point of evaluation episode 0:

| $t$ | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ball − paddle (px) | −79 | −53 | −43 | −32 | −26 | −36 | −50 | −44 | −20 | −10 |
| $P(\text{down})$ | 0.24 | 0.21 | 0.32 | 0.56 | 0.62 | 0.62 | 0.67 | 0.69 | 0.60 | 0.75 |
| Action | up | down | up | down | down | down | up | up | down | down |
| Reward | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | −1 |

- **[FAIL] The more likely action points towards the ball in only 3 of 10 steps.** The ball is above the paddle the whole time but moving down (its $y$ rises from 50 to 121). $P(\text{down})$ rises with it, from 0.24 to 0.75, so the policy follows the ball's motion, not its position: the same behaviour as in Check 4.
- The paddle therefore moves down with the ball instead of waiting for it, and the point is lost.

### Summary of 3b

| | Value |
|---|---|
| Training reward, first / last 100 episodes | −20.57 / −18.07 |
| Episode length, first / last 100 episodes | 993 / 2,065 steps |
| Evaluation (500 episodes, sampled) | **−17.66 ± 1.86** |
| Reference: random / ball tracker | −20.40 / −14.74 |
| Learned behaviour | Moves with the ball's vertical direction, not towards its position |

REINFORCE without a baseline learns Pong, but slowly: for the first ~700 of 2,000 episodes the policy does not react to the screen at all. It then learns the feature that is easiest to see in the frame difference, the ball's direction, and doubles the rally length. It still loses every game and was still improving when training ended.
