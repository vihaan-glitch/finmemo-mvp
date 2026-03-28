# FinMemo MVP

**Monte Carlo financial simulation with a safety constraint layer — generates confidence intervals instead of point estimates.**

Most financial memos report a single projected return: *"Expected ROI: +18%."* That number hides everything important — how wide the uncertainty is, how many scenarios actually lose money, and whether the headline figure is misleading given the underlying distribution.

FinMemo treats every input assumption as a distribution, runs thousands of simulated scenarios, and applies a structured safety layer before producing any output. If the uncertainty is too high, the memo is blocked. If it needs a disclaimer, one is added automatically.

---

## How It Works

### 1. Assumptions as distributions
Instead of a fixed growth rate, you provide a mean and standard deviation:

```python
growth = AssumptionRange("revenue_growth", mean=0.12, std=0.04)
discount = AssumptionRange("discount_rate", mean=0.10, std=0.02)
```

### 2. Monte Carlo simulation
The engine draws `n` samples from each distribution, computes a terminal-value NPV for each scenario, and returns the full result distribution — mean, median, 90% CI, and the percentage of scenarios that are positive.

```python
result = run_npv_simulation(
    initial_investment=1_000_000,
    revenue_growth=growth,
    discount_rate=discount,
    years=5,
    n_simulations=10_000,
)
# result.mean, result.p5, result.p95, result.positive_pct
```

### 3. Safety constraint layer
Before any output is published, `check_safety()` evaluates the result against configurable constraints and returns a `SafetyReport`:

| Condition | Severity | Action |
|---|---|---|
| CI width > 150% of mean | Hard | **Block** — too uncertain to publish |
| Mean positive but <40% of scenarios positive | Hard | **Block** — headline is misleading |
| High dispersion (σ / \|μ\| > 2×) | Soft | Add disclaimer |
| CI width > 80% of mean | Soft | Add disclaimer |

```python
safety = check_safety(result, SafetyConstraints())
# safety.recommended_action → "publish" | "add_disclaimer" | "flag_for_review" | "block"
```

This constraint design is inspired by constrained Markov Decision Processes (CMDPs), where a policy must satisfy hard safety constraints while optimizing an objective — applied here to financial document generation.

---

## Motivation

Standard financial models produce point estimates that overstate precision. The real question is not *"what is the expected return?"* but *"what is the probability distribution over returns, and is it safe to headline the mean?"*

This project is an attempt to formalize that question with a structured constraint layer, rather than leaving it to human judgment after the fact.

---

## Quickstart

**Requirements:** Python 3.11+

```bash
git clone https://github.com/vihaan-glitch/finmemo-mvp
cd finmemo-mvp
pip install -r requirements.txt
streamlit run main.py
```

The app opens in your browser. Adjust the sidebar inputs and observe how the safety layer responds as uncertainty increases.

---

## Project Structure

```
finmemo-mvp/
├── main.py              # Streamlit UI
├── core/
│   ├── monte_carlo.py   # AssumptionRange, MonteCarloResult, run_npv_simulation
│   └── safety.py        # SafetyConstraints, Violation, SafetyReport, check_safety
└── requirements.txt
```

---

## Current Limitations

- The financial model uses a simplified single terminal cash flow, not a full multi-period DCF. The research contribution is the uncertainty quantification and safety constraint framework, not the financial model itself.
- Correlation between input assumptions (e.g., growth and discount rate moving together) is not yet modeled.
- Safety thresholds in `SafetyConstraints` are heuristic defaults — calibration against real financial data is future work.

---

## Background

Built by Vihaan, a high school sophomore at Morris Hills High School (Morris Plains, NJ), alongside research in reinforcement learning and robotics with Professor Arnob Ghosh at NJIT. The safety constraint architecture draws on constrained MDP work from that research applied to a financial domain.
