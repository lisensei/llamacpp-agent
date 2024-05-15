from flask import Flask
from langchain_community.llms import LlamaCpp
from langchain.agents import create_react_agent,AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from argparse import ArgumentParser
import langchain
langchain.debug=True
parser=ArgumentParser()
parser.add_argument("-model_path",type=str)
parser.add_argument("-gpu_layers",type=int,default=-1)
parser.add_argument("-context_length",type=int,default=4096)
parser.add_argument("-temperature",type=float,default=1)
parser.add_argument("-max_tokens",type=int,default=4096)
parser.add_argument("-question",type=str)
args=parser.parse_args()
app=Flask(__name__)
app.config["model_path"]=args.model_path
bot="<|begin_of_text|>"
shi="<|start_header_id|>"
ehi="<|end_header_id|>"
eot="<|eot_id|>"
temp='''
{bot}{shi}system{ehi}You are an accurate intelligent system. You one and only task is answer any questions. You must use this tool: {tools}, to search for information. 
The name of the tool is: {tool_names}.

You should use the following pattern to answer questions:

Question: The question you must answer
Thought: The thinking process of answering the question
Action: The action needed to answer the question (it must only be the exact tool name, no extra characters allowed)
Action Input: The parameters passed to the action
Observation: The result of taking the action
Final Answer: The final answer to the question

The Thought-Action-Action Input-Observataion process can repeat at most 2 times. When you can't find the answer, return results not found.


Here are two examples:

Question: What's the meaning of life?
Thought: To answer this question, I should search the web using tavily_search_results_json
Action: tavily_search_results_json
Action Input: What's the meaning of life?
Observation: Based on the result returned from tavily_search_results_json, the answer to the question is 42.
Final Answer: The meaning of life is 42.

Question: What's Llama 3?
Thought: To find the answer to this question, I must use search the Internet using tavily_search_results_json
Action: tavily_search_results_json
Action Input: What's Llama3?
Observation:  After search the web using tavily_search_results_json, I now know llama 3 is a open source LLM.
Final Answer: Llama 3 is a open source large language model.

Remember, the tool's name is case sensitive, you must use this exact tool name: tavily_search_results_json

Start!
{eot}
{shi}intelligent system{ehi}
Question: {input}
{agent_scratchpad}
'''
model=LlamaCpp(model_path=app.config["model_path"],n_gpu_layers=args.gpu_layers,n_ctx=args.context_length,max_tokens=args.max_tokens,temperature=args.temperature)
tools=[TavilySearchResults(max_results=1)]
prompt=PromptTemplate.from_template(temp)
age=create_react_agent(model,tools,prompt)
agent=AgentExecutor(agent=age,tools=tools,verbose=True,handle_parsing_errors=True)
agent.invoke({"input":args.question,"bot":bot,"shi":shi,"ehi":ehi,"eot":eot})