import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class AssumptionRange:
    """
    A financial assumption as a distribution, not a single number.
    Instead of "growth rate = 20%", you say:
    "growth rate ~ Normal(mean=0.20, std=0.08)"
    """
    name: str
    mean: float
    std: float
    dist: Literal["normal", "uniform"] = "normal"

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.std < 0:
            raise ValueError(f"{self.name} std must be >= 0.")
        if n <= 0:
            raise ValueError("n must be > 0.")

        generator = rng or np.random.default_rng()
        if self.dist == "normal":
            return generator.normal(self.mean, self.std, n)
        low = self.mean - self.std
        high = self.mean + self.std
        return generator.uniform(low, high, n)


@dataclass
class MonteCarloResult:
    """Full distribution output from a simulation run."""
    values: np.ndarray      
    mean: float
    median: float
    std: float
    p5: float             
    p25: float
    p75: float
    p95: float             
    positive_pct: float  
    ci_width_pct: float   
    n_simulations: int

    @property
    def ci_90(self) -> tuple[float, float]:
        return (self.p5, self.p95)

    @property
    def summary_str(self) -> str:
        sign = "+" if self.mean >= 0 else ""
        return (
            f"{sign}{self.mean:.1f}%  "
            f"(90% CI: {self.p5:+.1f}% to {self.p95:+.1f}%,  "
            f"{self.positive_pct:.0f}% of scenarios positive)"
        )


def run_npv_simulation(
    initial_investment: float,
    revenue_growth: AssumptionRange,
    discount_rate: AssumptionRange,
    years: int = 5,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> MonteCarloResult:
    if initial_investment <= 0:
        raise ValueError("initial_investment must be > 0.")
    if years <= 0:
        raise ValueError("years must be > 0.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be > 0.")
    if revenue_growth.std < 0:
        raise ValueError("revenue_growth std must be >= 0.")
    if discount_rate.std < 0:
        raise ValueError("discount_rate std must be >= 0.")

    rng = np.random.default_rng(seed)
    growth_samples = revenue_growth.sample(n_simulations, rng=rng)  # shape: (n_sim,)
    rate_samples = discount_rate.sample(n_simulations, rng=rng)  # shape: (n_sim,)

    # Terminal-value NPV: growth compounds once to the horizon; discount that single
    # future value — do not sum intermediate (1+g)^t as if each year were a separate
    # cash inflow (that double-counts compounding and overstates returns).
    terminal_value = initial_investment * (1 + growth_samples) ** years
    pv_terminal = terminal_value / (1 + rate_samples) ** years

    # Express as % return on investment — easier to read than raw NPV dollars
    npv_values = pv_terminal - initial_investment
    irr_approx   = (npv_values / initial_investment) * 100   # shape: (n_sim,)

    mean = float(np.mean(irr_approx))
    p5   = float(np.percentile(irr_approx, 5))
    p95  = float(np.percentile(irr_approx, 95))

    return MonteCarloResult(
        values       = irr_approx,
        mean         = mean,
        median       = float(np.median(irr_approx)),
        std          = float(np.std(irr_approx)),
        p5           = p5,
        p25          = float(np.percentile(irr_approx, 25)),
        p75          = float(np.percentile(irr_approx, 75)),
        p95          = p95,
        positive_pct = float((irr_approx > 0).mean() * 100),
        ci_width_pct = float((p95 - p5) / max(abs(mean), 1e-6) * 100),
        n_simulations = n_simulations,
    )