from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

import os
os.environ["MISTRAL_API_KEY"] = ""

model="mistral-small-latest" 

llm=ChatMistralAI(
    temperature=0.0,
    model=model,
    max_retries=2,
)

# # LLMChain
prompt=ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")

chain=LLMChain(llm=llm,prompt=prompt)

product="Queen Size Sheet Set"
print(chain.run(product))

# SequenceChain
product="Queen Size Sheet Set"

# # Chain 1
prompt_one=ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain_one=LLMChain(llm=llm,prompt=prompt_one,output_key="company_name")

# Chain 2
prompt_two=ChatPromptTemplate.from_template("Write a 10 words description for the following company:{company_name}")
chain_two=LLMChain(llm=llm,prompt=prompt_two,output_key="description")

over_all_simple_chain=SequentialChain(
    chains=[chain_one,chain_two],
    input_variables=['product'],
    output_variables=['company_name','description']
)

result = over_all_simple_chain.invoke(product)
print(result)


# # Router Chain
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos=[
    {
        "name":"physics",
        "description": "Good for answering about physics",
        "prompt_template":physics_template
    },
    {
        "name":"math",
        "description": "Good for answering about math",
        "prompt_template":math_template
    },
    {
        "name":"history",
        "description": "Good for answering about history",
        "prompt_template":history_template
    },
    {
        "name":"computer science",
        "description": "Good for answering about computer science",
        "prompt_template":computerscience_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

destination_chains={}

for p_info in prompt_infos:
    name=p_info['name']
    prompt_template=p_info['prompt_template']
    prompt=ChatPromptTemplate.from_template(template=prompt_template)
    chain=LLMChain(llm=llm,prompt=prompt)
    destination_chains[name]=chain

destinations=[f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str="\n".join(destinations)

default_prompt=ChatPromptTemplate.from_template("{input}")
default_chain=LLMChain(llm=llm,prompt=prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""


router_template=MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str)

router_prompt=PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser()
)

router_chain=LLMRouterChain.from_llm(llm,router_prompt)

chain=MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=False
)
result=chain.invoke("What is black body radiation?")
print(result)