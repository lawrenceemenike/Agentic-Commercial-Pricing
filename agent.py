import pandas as pd
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from optimize_price import run_xgboost_optimization
import os

class PricingAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    target_month: int
    weight: float
    freight: float
    cogs: float
    market_context: dict
    optimization_results: dict

def extract_category_context(category: str, target_month: int, weight: float, freight: float, cogs: float, **kwargs) -> dict:
    """Extracts the market conditions (competitor prices, seasonality, etc) for a given category and month from the CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), "retail_price.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": "retail_price.csv not found."}
        
    cat_data = df[(df['product_category_name'] == category) & (df['month'] == target_month)]
    if cat_data.empty:
        return {"error": f"Category {category} and month {target_month} not found in data."}
    
    context = {
        'category': category,
        'target_month': target_month,
        'freight_price': float(freight),
        'product_weight_g': float(weight),
        'cogs': float(cogs),
        'comp_1': float(cat_data['comp_1'].median()),
        'comp_2': float(cat_data['comp_2'].median()),
        'comp_3': float(cat_data['comp_3'].median()),
        'month': int(target_month),
        'holiday': int(cat_data['holiday'].mode()[0]),
        'weekend': int(cat_data['weekend'].median()),
        's': float(cat_data['s'].median()),
        'lag_price': float(cat_data['lag_price'].median()),
        'product_score': float(cat_data['product_score'].median()),
        'unit_price': float(cat_data['unit_price'].median())
    }
    
    results = run_xgboost_optimization(context)
    return results


import time
from datetime import datetime
import streamlit as st
from agentmesh.governance import govern

# 1. Apply governance to raw Python functions
governed_extract = govern(extract_category_context, policy="governance/policy.yaml")

# 2. Wrap the governed callables back into LangChain Tools to support native `.invoke()` and schema extraction
@tool
def extract_category_context_tool(category: str, target_month: int, weight: float, freight: float, cogs: float) -> dict:
    """Extracts market conditions and runs the XGBoost pricing simulation."""
    return governed_extract(action={"tool_name": "extract_category_context"}, category=category, target_month=target_month, weight=weight, freight=freight, cogs=cogs)

# Use llama3.2 via Ollama
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools([extract_category_context_tool])

def router_node(state: PricingAgentState):
    messages = state.get('messages', [])
    
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        user_text = last_msg.content.lower()
        if any(word in user_text for word in ['delete', 'drop', 'write', 'scrape', 'hack', 'install', 'execute']):
            rejection = "I am a governed Commercial Intelligence Engine. I do not have authorization or system-level access to execute that command."
            return {"messages": [AIMessage(content=rejection)]}

    EXECUTIVE_SYSTEM_PROMPT = """You are a Lead Commercial Finance Strategist advising the C-suite. You have over 20 years experience in commerical and pricing strategy, supply chain and pricing optimization, geopolitics and business strategy. You will receive a JSON payload containing the results of a pricing optimization simulation. 

YOUR DIRECTIVES:
CRITICAL DIRECTIVE: You currently have NO pricing data. To answer the user, you MUST execute the extract_category_context_tool to run the XGBoost simulation. You are forbidden from answering without running the tool first.

CONVERSATIONAL MEMORY RULE: You may only use memory for follow-up questions regarding the EXACT SAME category and month. If the user mentions a NEW category, a NEW month, or says they updated the dashboard, your previous memory is STALE and INVALID. You are STRICTLY FORBIDDEN from reusing old numbers. You MUST execute the extract_category_context_tool with the new parameters.

EXPECTED OUTPUT FORMAT:
"The intelligence engine recommends an optimal price of $[optimal_price]. At this price point, we secure a Gross Margin of [gross_margin_pct]% while maintaining a [comp_premium_pct]% position relative to the primary competitor ($[primary_competitor_price])."

1. Start your response directly with the exact EXPECTED OUTPUT FORMAT provided above.
2. You MUST use the exact data provided in the tool output. Do not calculate these yourself.
3. You will receive a list of 'key_drivers' in the payload. These are the top 3 market factors that statistically drove the algorithm's elasticity forecast. 
4. Write a 3-4 sentence strategic justification explaining how our recommended price positions us against the primary competitor ('comp_1') while leveraging the 'key_drivers'.
5. SCENARIO 2: UNPROFITABLE MARKET (ERROR)
Inform the executive that the market is structurally unprofitable because our combined COGS and Freight costs create a break-even floor higher than the market will pay. You will receive 'key_drivers' in the error payload. You may mention that "Despite strong demand signals from [key_drivers], our unit economics prevent market entry." You are STRICTLY FORBIDDEN from inventing or hallucinating key drivers.
6. Do not output raw JSON, dictionaries, or tool execution logs. Format your response as a clean, authoritative executive paragraph.
"""
    
    sys_msg = SystemMessage(content=EXECUTIVE_SYSTEM_PROMPT)
    
    # Strictly enforce a single, fresh SystemMessage at the start of the state list
    clean_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    messages = [sys_msg] + clean_messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: PricingAgentState):
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    tool_messages = []
    market_ctx = state.get('market_context', {})
    opt_results = state.get('optimization_results', {})
    category = state.get('category', "")
    target_month = state.get('target_month', 1)
    weight = state.get('weight', 0.0)
    freight = state.get('freight', 0.0)
    cogs = state.get('cogs', 40.0)
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            start_time = time.time()
            result_status = "Allow"
            res = None
            
            if tool_call['name'] == 'extract_category_context_tool':
                # Force override LLM hallucinated args with actual state values to prevent hallucination loop
                args = {
                    "category": category,
                    "target_month": target_month,
                    "weight": weight,
                    "freight": freight,
                    "cogs": cogs
                }
                try:
                    res = extract_category_context_tool.invoke(args)
                except Exception as e:
                    if "governance" in str(e).lower() or "policy" in str(e).lower():
                        res = {"error": "Action blocked by enterprise governance policy."}
                        result_status = "Deny"
                    else:
                        raise e
                
                tool_messages.append(ToolMessage(content=json.dumps(res), tool_call_id=tool_call['id']))
                if isinstance(res, dict) and "error" not in res:
                    opt_results = res
            else:
                tool_messages.append(ToolMessage(content="Unknown tool", tool_call_id=tool_call['id']))
                
            latency = time.time() - start_time
            if 'audit_log' in st.session_state:
                st.session_state.audit_log.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tool_requested": tool_call['name'],
                    "policy_evaluated": "pricing-terminal-production-policy",
                    "latency": latency,
                    "result": result_status
                })
                
    return {
        "messages": tool_messages,
        "market_context": market_ctx,
        "optimization_results": opt_results,
        "category": category,
        "target_month": target_month,
        "weight": weight,
        "freight": freight,
        "cogs": cogs
    }

def should_continue(state: PricingAgentState):
    messages = state.get('messages', [])
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(PricingAgentState)
workflow.add_node("agent", router_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

compiled_graph = workflow.compile()
