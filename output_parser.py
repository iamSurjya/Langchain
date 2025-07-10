customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

review_template2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.
text: {text}
{format_instruction}
"""

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import os
os.environ["MISTRAL_API_KEY"] = ""

model="mistral-small-latest" #"mistral-large-latest"
llm = ChatMistralAI(
    model=model,
    temperature=0,
    max_retries=2,
)

# without Output parser
# prompt_template=ChatPromptTemplate.from_template(review_template)
# messages=prompt_template.format_messages(text=customer_review)

# response=llm.invoke(messages)
# print(type(response.content)) # this return strings
# print(response.content)

# with Ouput parser
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema=ResponseSchema(name="gift",description="gift")
delivery_schema=ResponseSchema(name="delivery_days",description="number of days for delivery")
price_value=ResponseSchema(name="price_value",description="price_value")

response_schemas=[gift_schema,delivery_schema,price_value]
output_parser=StructuredOutputParser.from_response_schemas(response_schemas)

format_instruction=output_parser.get_format_instructions()

print(format_instruction)

prompt=ChatPromptTemplate.from_template(template=review_template2)
messages=prompt.format_messages(text=customer_review,
                                format_instruction=format_instruction)

response=llm.invoke(messages)

# Parse the response string into a dict
parsed_output = output_parser.parse(response.content)
print(parsed_output)             # 
print(type(parsed_output))       # returns <class 'dict'>