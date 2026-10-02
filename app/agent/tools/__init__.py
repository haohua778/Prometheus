"""Tool registry. Each capability Prometheus gains is one tool module here, added to TOOLS."""
from langchain_core.tools import BaseTool

TOOLS: list[BaseTool] = []
