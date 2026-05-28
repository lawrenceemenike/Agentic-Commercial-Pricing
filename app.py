import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage, AIMessage
from agent import compiled_graph
import os
import json

st.set_page_config(layout="wide", page_title="Commercial Intelligence Terminal")

st.title("Agentic Commercial Pricing Terminal")

left_col, right_col = st.columns(2)


if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": [], "market_context": {}, "optimization_results": {}, "category": "", "target_month": 1, "weight": 0.0, "freight": 0.0, "cogs": 0.0}
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

with left_col:
    st.header("Commercial Intelligence KPIs")
    
    csv_path = os.path.join(os.path.dirname(__file__), "retail_price.csv")
    df = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.warning("retail_price.csv not found. Please run generate_mock_data.py first.")

    if df is not None:
        st.subheader("Market Overview")
        
        categories = df['product_category_name'].unique()
        
        with st.form(key='simulation_form'):
            st.markdown("### Simulation Parameters")
            selected_cat = st.selectbox("Category", categories)
            target_month = st.selectbox("Target Month", list(range(1, 13)))
            
            cat_df = df[df['product_category_name'] == selected_cat]
            default_weight = float(cat_df['product_weight_g'].median())
            default_freight = float(cat_df['freight_price'].median())
            
            weight = st.number_input("Product Weight (g)", value=default_weight)
            freight = st.number_input("Expected Freight Cost ($)", value=default_freight)
            cogs = st.number_input("Base Cost of Goods (COGS) ($)", value=40.0)
            
            submit_button = st.form_submit_button('Update Market Context')
            
        if submit_button:
            st.session_state.agent_state['category'] = selected_cat
            st.session_state.agent_state['target_month'] = target_month
            st.session_state.agent_state['weight'] = weight
            st.session_state.agent_state['freight'] = freight
            st.session_state.agent_state['cogs'] = cogs
            st.session_state.agent_state['messages'] = []
            st.session_state.audit_log = []
            st.toast("Commercial Context Updated. Chat memory wiped for new simulation.")
            st.success(f"Market context updated to {selected_cat} for Month {target_month}. You can now prompt the agent.")
            
        context_df = df[(df['product_category_name'] == selected_cat) & (df['month'] == target_month)]
        
        if not context_df.empty:
            cat_avg_price = context_df['unit_price'].median()
            peak_seasonality = context_df['s'].median() 
            market_leader_avg = context_df['comp_1'].median()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Category Average Price", f"${cat_avg_price:.2f}")
            col2.metric("Peak Seasonality", f"{peak_seasonality:.2f}")
            col3.metric("Market Leader Avg", f"${market_leader_avg:.2f}")

        st.subheader("Price vs. Predicted Demand (Historical)")
        
        labels_map = {
            "unit_price": "Our Unit Price ($)",
            "qty": "Predicted Demand / Sales Volume",
            "comp_1": "Primary Competitor Price",
            "fp1": "Primary Competitor Freight",
            "ps1": "Primary Competitor Rating",
            "s": "Seasonality Index"
        }
        fig = px.scatter(cat_df, x="unit_price", y="qty", color="month", title=f"Historical Price vs Demand for {selected_cat}", labels=labels_map)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.agent_state.get("optimization_results"):
            st.success("Recent Optimization Run Results:")
            results = st.session_state.agent_state["optimization_results"]
            st.write(f"**Optimal Price:** ${results.get('optimal_price', 0):.2f}")
            st.write(f"**Expected Volume:** {results.get('expected_volume', 0):.0f}")
            st.write(f"**Projected Profit:** ${results.get('projected_profit', 0):.2f}")

with right_col:
    st.header("Agentic Copilot")
    
    # Render chat history
    for msg in st.session_state.agent_state["messages"]:
        if msg.type in ["human", "user"]:
            st.chat_message("user").write(msg.content)
        elif msg.type in ["ai", "assistant"]:
            # Check if content is a string. If it's a list (from tool calls), extract the text.
            if isinstance(msg.content, str):
                if msg.content.strip():
                    safe_text = msg.content.replace('$', '\\$')
                    st.chat_message("assistant").write(safe_text)
            elif isinstance(msg.content, list) and len(msg.content) > 0 and 'text' in msg.content[0]:
                safe_text = msg.content[0]['text'].replace('$', '\\$')
                st.chat_message("assistant").write(safe_text)
            
    # Chat input
    if prompt := st.chat_input("What is the optimal price for this category in the target month?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing market conditions and running simulations..."):
                try:
                    # Append HumanMessage to agent state
                    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))
                    
                    # Run the graph
                    final_state = compiled_graph.invoke(st.session_state.agent_state)
                    
                    # Update session agent state
                    st.session_state.agent_state = final_state
                    
                    # Get last message
                    last_msg = final_state["messages"][-1]
                    
                    if isinstance(last_msg.content, str) and last_msg.content.strip():
                        safe_text = last_msg.content.replace('$', '\\$')
                        st.markdown(safe_text)
                    elif isinstance(last_msg.content, list) and len(last_msg.content) > 0 and 'text' in last_msg.content[0]:
                        safe_text = last_msg.content[0]['text'].replace('$', '\\$')
                        st.markdown(safe_text)
                        
                    st.rerun() # Trigger rerun to update the left column if new optimization happened
                except Exception as e:
                    st.error(f"Error during graph execution: {e}")
            
    with st.expander("🛡️ Decision Bill of Materials & Audit Log"):
        if not st.session_state.audit_log:
            st.write("No agent actions recorded yet.")
        else:
            for log in reversed(st.session_state.audit_log):
                status_icon = "✅" if log['result'] == "Allow" else "❌"
                st.markdown(f"**{status_icon} {log['timestamp']}** | {log['tool_requested']}")
                st.markdown(f"- **Policy Evaluated:** {log['policy_evaluated']}")
                st.markdown(f"- **Latency:** {log['latency']:.4f}s | **Result:** {log['result']}")
                st.divider()
