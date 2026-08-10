"""
backend.agents — LangChain / LangGraph agent definitions.

Agents use LangGraph ``StateGraph`` to model multi-step reasoning workflows.
Each agent module exposes a compiled graph that can be invoked from a service.

Planned agents:
- ``recommendation_agent`` : Personalised product recommendations.
- ``support_agent``        : Customer support Q&A with RAG.
- ``analytics_agent``      : Natural-language queries over sales data.
"""
