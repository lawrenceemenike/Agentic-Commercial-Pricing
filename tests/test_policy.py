import pytest
import os
from agentmesh.governance import PolicyEvaluator
from agentmesh.exceptions import GovernanceError

def test_policy_allows_whitelisted_tools():
    evaluator = PolicyEvaluator("governance/policy.yaml")
    
    # Should not raise exceptions
    assert evaluator.evaluate(action_type="tool_execution", tool_name="extract_market_conditions") == "allow"
    assert evaluator.evaluate(action_type="tool_execution", tool_name="run_xgboost_optimization_tool") == "allow"

def test_policy_denies_destructive_actions():
    evaluator = PolicyEvaluator("governance/policy.yaml")
    
    with pytest.raises(GovernanceError):
        evaluator.evaluate(action_type="network_request")
        
    with pytest.raises(GovernanceError):
        evaluator.evaluate(action_type="delete")
