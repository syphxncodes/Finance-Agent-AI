from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os

#Web Agent 
websearch_agent=Agent(
    name="Web Search Agent",
    role="Be the agent you always wanted to be, and become a search analyst for me about the query that I ask.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=True,
    markdown=True,
)
# Financial Agent
finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True,company_info=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)
# combines both the agents.
multi_model_ai_agent=Agent(
    team=[websearch_agent,finance_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Always include sources and use the tables to display the data!"],
    show_tool_call=True,
    markdown=True
)
multi_model_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA.",stream=True)