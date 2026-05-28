from agent.py import extract_market_conditions
from agentmesh.governance import govern

governed_extract = govern(extract_market_conditions, policy="governance/policy.yaml")
print("Type:", type(governed_extract))
print("Dir:", dir(governed_extract))
if hasattr(governed_extract, 'invoke'):
    print("Has invoke")
else:
    print("No invoke")
