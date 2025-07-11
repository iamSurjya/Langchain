from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

import os
os.environ["MISTRAL_API_KEY"] = ""

model="mistral-small-latest" 

llm=ChatMistralAI(
    temperature=0.0,
    model=model,
    max_retries=2,
)

# ConversationBufferMemory
memory=ConversationBufferMemory()

conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi,My Name is Spruha")
conversation.predict(input="What is 1+1")
conversation.predict(input="What is my name")

print(memory.buffer)

# ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=1)

conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi,My Name is Spruha")
conversation.predict(input="What is 1+1")
conversation.predict(input="What is my name")

print(memory.buffer)

# ConversationTokenBufferMemory
memory=ConversationTokenBufferMemory(llm=llm,max_token_limit=10)

conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi,My Name is Spruha")
conversation.predict(input="What is 1+1")
conversation.predict(input="What is my name")

print(memory.buffer)

# ConversationSummaryMemory
memory=ConversationSummaryBufferMemory(llm=llm,max_token_limit=10)

conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi,My Name is Spruha")
conversation.predict(input="What is 1+1")
conversation.predict(input="What is my name")

print(memory.buffer)