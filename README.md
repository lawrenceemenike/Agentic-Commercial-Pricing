# Agentic Commercial Pricing Terminal

The **Agentic Commercial Pricing Terminal** is an advanced, AI-driven commercial intelligence platform. It fuses deterministic machine learning (XGBoost) with a cognitive reasoning layer (Llama 3.2 via Ollama) orchestrated by LangGraph to dynamically calculate and strategically justify optimal pricing for retail categories.

## 🚀 Architecture Overview

This platform features a dual-layer architecture designed to prevent AI hallucinations while delivering strategic, executive-level insights.

1. **Predictive Engine (XGBoost):**
   - Located in `optimize_price.py`.
   - Simulates price elasticity across a bounded commercial matrix.
   - Evaluates historical seasonality, primary competitor pricing, freight costs, and COGS.
   - Outputs deterministic, mathematically sound pricing recommendations, margins, and key drivers.

2. **Cognitive Reasoning Layer (Llama 3.2):**
   - Located in `agent.py` and orchestrated using LangGraph.
   - Operates strictly as a "Commercial Finance Strategist."
   - Executed via an "Anti-Cheat Blindfold"—the LLM is structurally forbidden from accessing raw data directly, forcing it to invoke the XGBoost tool to retrieve mathematically sound outputs.
   - Synthesizes the optimized numbers into an executive summary without attempting to perform math itself.

3. **Enterprise Governance (AgentMesh):**
   - Enforces Zero-Trust tool execution via `policy.yaml`.
   - Prevents unauthorized commands, data deletion, or unauthorized system access during agentic reasoning loops.

4. **Interactive Dashboard (Streamlit):**
   - Located in `app.py`.
   - Provides a real-time UI for commercial managers to set Cost of Goods Sold (COGS), freight expectations, and product weights.
   - Features robust state invalidation to prevent cross-category data bleed.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed locally with the `llama3.2` model downloaded (`ollama run llama3.2`).

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lawrenceemenike/Agentic-Commercial-Pricing.git
   cd Agentic-Commercial-Pricing
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the Synthetic Market Data:**
   ```bash
   python generate_mock_data.py
   ```
   *This will generate a highly correlated, multi-category `retail_price.csv` dataset for the engine to train on.*

5. **Launch the Terminal:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Core Features & Safeguards

- **Dynamic Margin Calculation:** Break-even floors are dynamically calculated based on user-defined COGS and Freight. The XGBoost engine is mathematically blocked from simulating prices that yield a negative margin.
- **Elasticity Penalty:** Simulates realistic demand destruction by penalizing projected volume by 2% for every 1% the price scales above the primary market leader.
- **Context Invalidation:** Changing the category or month in the UI completely flushes the LangGraph state memory, preventing the LLM from hallucinating data across distinct product categories.
- **Error Grounding:** In the event of an unprofitable market state, the LLM is explicitly barred from hallucinating reasons and must instead report actual demand drivers causing the failure.

## 📜 License
MIT License
