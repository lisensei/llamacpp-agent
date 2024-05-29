# Llama ReAct Search Agent

This is a search agent using llama3 and ReAct prompting.

Groq cloud api key (if using Groq as LLM provider) and Tavily api key are required. 

# Usage:
First, run the api by:

python api.py

Then:

streamlit run gui.py

# Example Search
![llamacpp-agent](assets/agent_demo.gif)

Note that Llama 3's knowledge cutoff date is before May, 2024. The agent found the correct the answer.