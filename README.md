This project is the second Chapter of my PhD Thesis in quantitative finance.

This paper provides a conceptual and computational roadmap for heterogeneous-agent models in macro-finance, with a sharp focus on what matters for **asset pricing, systemic risk, and quantitative implementation**.

It argues that the dominant modeling paradigms — few-agent models and mean-field games — each miss something essential for finance, and it proposes granular N-player models as the natural middle ground.

---

## 🧭 Why This Matters 

Most finance models rely on:

- 2–3 agent setups (borrower vs. saver, optimist vs. pessimist, intermediary vs. household)
- Or continuum-agent models where individuals are infinitesimal

The paper shows:

- **Few-agent models are too coarse** → idiosyncratic risk is effectively aggregate risk.
- **Mean-field games are too fine** → idiosyncratic shocks wash out and cannot generate aggregate risk.
- **Finance requires endogenous aggregate risk**.

The paper frames heterogeneous-agent modeling not as a distributional exercise — but as a way to model the *microfoundations of risk premia*.

---

## ⚖️ The Core Modeling Trade-Off

### 🔹 Mean-Field Games 

**Strengths**
- Rich cross-sectional heterogeneity
- Tractable for distributional dynamics
- Powerful for inequality and policy analysis

**Fatal limitation for finance**
- Agents are infinitesimal
- Idiosyncratic shocks cancel out
- Aggregate risk must be imposed exogenously

Implication:
- Risk premia are taken as given.
- The model cannot explain where volatility comes from.

For asset pricing, this is a structural limitation.

---

### 🔹 Granular N-Player Games

Inspired by granular origins (e.g. large-firm effects), these models feature:

- Many agents
- Heterogeneous sizes
- Non-negligible individual impact

Now:

- Large intermediaries can transmit idiosyncratic shocks into aggregate outcomes
- The size distribution itself becomes a state variable
- Aggregate risk becomes endogenous

For quantitative finance, this opens new modeling possibilities:

- Concentration risk
- Market maker fragility
- Intermediary leverage cycles
- Order-flow driven volatility
- Endogenous liquidity crises

This framework connects directly to how modern financial markets actually function: dominated by a fat-tailed distribution of institutions.

---

## 📊 Modeling Order Flow and Market Ecology

Modern markets are not representative-agent systems. They consist of:

- Hedge funds
- Pension funds
- Banks
- Broker-dealers
- Insurance companies
- Retail traders
- High-frequency market makers

Granular heterogeneous-agent models allow you to:

- Model the size distribution of participants
- Track wealth reallocation across strategies
- Generate endogenous volatility from concentration
- Study how large players affect price impact

This is directly relevant for:

- Flow-based return predictability
- Liquidity provision modeling
- Stress-testing dealer balance sheets
- Understanding volatility clustering
- Modeling cross-sectional risk transmission

---

## 📉 Endogenous Risk Premia

The paper emphasizes a key point:

> Risk premia require aggregate uncertainty, and aggregate uncertainty must come from somewhere.

Granular models provide:

- Microfoundations for time-varying risk premia
- A mechanism for countercyclical price of risk
- Wealth redistribution across heterogeneous agents
- Size-driven amplification of shocks

For quantitative asset allocation:

- The identity of the marginal investor changes over time
- Risk aversion becomes state-dependent
- The distribution of exposures matters

---

## 🏦 Systemic Risk & Concentration

Two-agent models cannot distinguish between:

- A global systemically important bank
- A regional bank

Granular N-player models can:

- Encode size distributions
- Model endogenous systemic importance
- Capture amplification from concentration
- Study macroprudential policy under realistic heterogeneity

This is highly relevant for:

- Risk parity under stress
- Bank-intermediated markets
- Funding liquidity spirals
- Cross-asset contagion modeling

---

## 🧠 Computational Contribution: Deep Learning & Solution Methods

The second major contribution is methodological.

The paper evaluates modern deep learning approaches for solving high-dimensional heterogeneous-agent models.

### ❌ Naïve Physics-Informed Neural Networks (PINNs)

Commonly proposed in macro literature.

Problem in finance:

- Portfolio choice is a **diffusion control** problem.
- Requires accurate second derivatives (Hessian).
- PINNs struggle with value-function curvature.
- Highly sensitive to shape violations.
- Mini-batching often fails.

Empirical finding:
- Naïve PINNs fail even in the Merton portfolio problem at extreme wealth levels.

This is highly relevant for:

- Continuous-time portfolio optimization
- Stochastic control in asset pricing
- Dynamic leverage problems

---

### ✅ Actor-Critic as Robust Alternative

The paper advocates:

- Separate networks for value and policy functions
- Avoid direct reliance on value function curvature
- More stable training
- Better scalability

For quant researchers:

- Actor-critic is more promising for solving high-dimensional equilibrium models
- Bridges reinforcement learning and macro-finance

This opens research avenues in:

- Deep RL for general equilibrium
- Learning-based equilibrium solvers
- Agent-based financial equilibrium models

---

## ⚙️ Conceptual Challenge: Rational Expectations

Solving heterogeneous-agent models with aggregate risk requires solving the “Master Equation” — the so-called “Monster Equation”.

Under rational expectations:

- Agents must forecast the entire future distribution.
- Computationally explosive.
- Conceptually implausible.

The paper discusses alternatives:

- Heuristic forecasting rules
- Survey-based expectations
- Adaptive learning
- Reinforcement learning

For financial markets, where beliefs and learning matter:

- This connects naturally to heterogeneous beliefs
- Flow-based pricing
- Experience-based learning
- Endogenous boom-bust dynamics

---

## 🚀 Research Directions Relevant for Quant Finance

Potential applications building on this framework:

- Endogenous volatility from heterogeneous leverage cycles
- Modeling dealer balance-sheet constraints with size distributions
- Flow-based factor models derived from heterogeneous-agent equilibrium
- Cross-asset contagion via granular intermediaries
- Reinforcement learning-based equilibrium solvers for asset pricing

---
