from dataclasses import dataclass, field
from core.monte_carlo import MonteCarloResult


@dataclass
class SafetyConstraints:
    max_ci_width_pct: float = 150.0      # hard block above this CI width
    min_positive_prob: float = 0.40      # block positive headline below this
    max_std_to_mean: float = 2.0         # soft flag for high dispersion
    # Floor for |mean| (percentage points) when computing σ/|μ|. Using a tiny epsilon
    # makes σ/μ explode near break-even; a modest floor treats near-zero mean as
    # "no meaningful relative scale" without conflating it with extreme risk.
    dispersion_mean_floor_pct: float = 5.0
    require_disclaimer_above_ci: float = 80.0  # auto-add disclaimer above this


@dataclass
class Violation:
    rule: str
    severity: str   # "hard" or "soft"
    message: str


@dataclass
class SafetyReport:
    safe: bool
    violations: list[Violation] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    recommended_action: str = "publish"  # publish | add_disclaimer | flag_for_review | block

    @property
    def hard_violations(self):
        return [v for v in self.violations if v.severity == "hard"]

    @property
    def soft_violations(self):
        return [v for v in self.violations if v.severity == "soft"]


def check_safety(
    result: MonteCarloResult,
    constraints: SafetyConstraints | None = None,
) -> SafetyReport:
    if constraints is None:
        constraints = SafetyConstraints()

    violations: list[Violation] = []
    disclaimers: list[str] = []

    # Hard: CI too wide
    if result.ci_width_pct > constraints.max_ci_width_pct:
        violations.append(Violation(
            rule="ci_width", severity="hard",
            message=(
                f"CI width {result.ci_width_pct:.0f}% of mean exceeds "
                f"limit of {constraints.max_ci_width_pct:.0f}%. Too uncertain to publish."
            )
        ))

    # Hard: misleading positive headline
    if result.mean > 0 and result.positive_pct < constraints.min_positive_prob * 100:
        violations.append(Violation(
            rule="positive_headline_mismatch", severity="hard",
            message=(
                f"Mean is positive but only {result.positive_pct:.0f}% of scenarios "
                f"are positive (threshold: {constraints.min_positive_prob*100:.0f}%)."
            )
        ))

    # Soft: high dispersion (σ vs |μ|, with floor on |μ| in % points to avoid σ/μ
    # blow-ups when mean is near zero / break-even)
    denom = max(abs(result.mean), constraints.dispersion_mean_floor_pct)
    std_to_mean = result.std / denom
    if std_to_mean > constraints.max_std_to_mean:
        violations.append(Violation(
            rule="high_dispersion", severity="soft",
            message=(
                f"High dispersion: σ / max(|μ|, {constraints.dispersion_mean_floor_pct:.0f}%) "
                f"= {std_to_mean:.1f}x."
            ),
        ))
        disclaimers.append(
            "Projections are highly sensitive to input assumptions. "
            "Outcomes vary significantly across scenarios."
        )

    # Soft: wide CI auto-disclaimer (must record a soft violation so recommended_action
    # becomes add_disclaimer, not publish, when only this rule fires)
    if constraints.require_disclaimer_above_ci < result.ci_width_pct <= constraints.max_ci_width_pct:
        violations.append(
            Violation(
                rule="wide_ci_disclaimer",
                severity="soft",
                message=(
                    f"90% CI width is {result.ci_width_pct:.0f}% of mean "
                    f"(above {constraints.require_disclaimer_above_ci:.0f}% disclosure threshold)."
                ),
            )
        )
        disclaimers.append(
            f"90% CI spans {result.p5:+.1f}% to {result.p95:+.1f}%. "
            "Consider the full range of outcomes."
        )

    hard = [v for v in violations if v.severity == "hard"]
    soft = [v for v in violations if v.severity == "soft"]

    if hard:
        action = "block"
    elif soft and disclaimers:
        action = "add_disclaimer"
    elif soft:
        action = "flag_for_review"
    else:
        action = "publish"

    return SafetyReport(
        safe=len(hard) == 0,
        violations=violations,
        disclaimers=disclaimers,
        recommended_action=action,
    )