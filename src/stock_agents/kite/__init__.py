"""Deterministic Kite Connect trading layer.

Deliberately has no dependency on agent_framework or the LLM crew - signal
generation, sizing and order placement must be backtestable and reproducible,
which rules out an LLM anywhere in this path. Use the research agents
(stock_agents.agents) for overnight preparation; use this module for execution.
"""
