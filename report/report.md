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
