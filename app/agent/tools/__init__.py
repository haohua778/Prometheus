"""Tool registry. Each capability Charon gains is one tool module here, added to TOOLS."""
from langchain_core.tools import BaseTool

TOOLS: list[BaseTool] = []
