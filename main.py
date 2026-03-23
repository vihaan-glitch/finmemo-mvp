import matplotlib.pyplot as plt
import streamlit as st

from core.monte_carlo import AssumptionRange, run_npv_simulation
from core.safety import SafetyConstraints, check_safety


def app() -> None:
    st.set_page_config(page_title="Safe Financial Memo", layout="wide")
    st.title("Safe Financial Memo")
    st.caption("Monte Carlo ROI simulation with safety checks.")

    with st.sidebar:
        st.header("Assumptions")
        investment = st.number_input(
            "Initial investment ($)",
            min_value=1_000.0,
            value=1_000_000.0,
            step=50_000.0,
        )
        years = st.slider(
            "Years",
            min_value=1,
            max_value=10,
            value=5,
            help="Horizon for the projection. Shown returns are cumulative over this period (approximate % on investment).",
        )
        n_simulations = st.slider(
            "Simulations",
            min_value=1_000,
            max_value=50_000,
            value=10_000,
            step=1_000,
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

        st.subheader("Revenue growth")
        growth_mean = (
            st.slider(
                "Mean growth (%)",
                min_value=-20.0,
                max_value=50.0,
                value=12.0,
                step=1.0,
                key="growth_mean_pct",
            )
            / 100
        )
        growth_std = (
            st.slider(
                "Growth std (%)",
                min_value=0.0,
                max_value=50.0,
                value=4.0,
                step=1.0,
                key="growth_std_pct",
            )
            / 100
        )

        st.subheader("Discount rate")
        discount_mean = (
            st.slider(
                "Mean discount (%)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.5,
                key="discount_mean_pct",
            )
            / 100
        )
        discount_std = (
            st.slider(
                "Discount std (%)",
                min_value=0.0,
                max_value=15.0,
                value=2.0,
                step=0.5,
                key="discount_std_pct",
            )
            / 100
        )

        if st.button("Load high-uncertainty scenario"):
            st.session_state["growth_mean_pct"] = 25.0
            st.session_state["growth_std_pct"] = 40.0
            st.session_state["discount_mean_pct"] = 10.0
            st.session_state["discount_std_pct"] = 8.0
            st.rerun()

    growth = AssumptionRange("revenue_growth", mean=growth_mean, std=growth_std)
    discount = AssumptionRange("discount_rate", mean=discount_mean, std=discount_std)

    try:
        result = run_npv_simulation(
            initial_investment=investment,
            revenue_growth=growth,
            discount_rate=discount,
            years=years,
            n_simulations=n_simulations,
            seed=int(seed),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    safety = check_safety(result, SafetyConstraints())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean ROI", f"{result.mean:+.1f}%")
    col2.metric("Median ROI", f"{result.median:+.1f}%")
    col3.metric("90% CI lower", f"{result.p5:+.1f}%")
    col4.metric("90% CI upper", f"{result.p95:+.1f}%")
    col5.metric("Positive scenarios", f"{result.positive_pct:.0f}%")
    st.caption(
        "Returns expressed as cumulative % ROI over the projection period, not annualized."
    )

    st.subheader("Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(result.values, bins=80, color="#378ADD", alpha=0.8, edgecolor="none")

    ci_mask = (result.values >= result.p5) & (result.values <= result.p95)
    ax.hist(
        result.values[ci_mask],
        bins=80,
        color="#185FA5",
        alpha=0.6,
        edgecolor="none",
        label=f"90% CI: {result.p5:+.1f}% to {result.p95:+.1f}%",
    )

    ax.axvline(result.mean, color="white", linewidth=2, label=f"Mean: {result.mean:+.1f}%")
    ax.axvline(
        0,
        color="#E24B4A",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Break-even",
    )

    ax.set_xlabel("Projected ROI (%)", fontsize=12)
    ax.set_ylabel("Number of scenarios", fontsize=12)
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor("#2a2a2a")
    legend.get_frame().set_edgecolor("#444")
    for text in legend.get_texts():
        text.set_color("white")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Safety")
    if not safety.safe:
        st.error(f"FAIL - {safety.recommended_action.upper()}")
    elif safety.recommended_action in ("add_disclaimer", "flag_for_review"):
        st.warning(f"PASS - {safety.recommended_action.upper()}")
    else:
        st.success(f"PASS - {safety.recommended_action.upper()}")

    for violation in safety.violations:
        label = violation.severity.upper()
        if violation.severity == "hard":
            st.error(f"[{label}] {violation.message}")
        else:
            st.warning(f"[{label}] {violation.message}")

    for disclaimer in safety.disclaimers:
        st.info(f"Disclaimer: {disclaimer}")

    st.subheader("Memo output")
    if safety.recommended_action == "block":
        st.error(
            "**[BLOCKED]** This estimate cannot be included. Reduce input uncertainty and re-run."
        )
    elif safety.recommended_action == "add_disclaimer":
        st.warning(
            f"Projected return: **{result.mean:+.1f}%** "
            f"(90% CI: {result.p5:+.1f}% – {result.p95:+.1f}%, "
            f"{result.positive_pct:.0f}% of scenarios positive). "
            + " ".join(safety.disclaimers)
        )
    else:
        st.success(
            f"Projected return: **{result.mean:+.1f}%** "
            f"(90% CI: {result.p5:+.1f}% – {result.p95:+.1f}%, "
            f"{result.positive_pct:.0f}% of scenarios positive)."
        )


if __name__ == "__main__":
    app()
