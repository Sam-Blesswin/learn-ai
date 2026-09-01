"""
LangChain Core Concepts - LCEL and Runnables
"""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are advertising expert. Create a tagline targeting {audience} for the following product: {product}. Be creative and concise."
    )

    model = ChatGroq(model="openai/gpt-oss-120b",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"product":" AI Course", "audience": "developers"})
    print(f"Response: {result}")

    return chain

def demo_basic_streaming():
    """Demonstrates a basic streaming chain using LCEL and Runnables."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in five sentence: {question}"
    )
    # model = ChatGroq(model="openai/gpt-oss-120b",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    parser = StrOutputParser()
    chain = prompt | model | parser

    for chunk in chain.stream({"question": "What is LangChain?"}):
        print(chunk, end="", flush=True)
    return chain

def demo_multi_chat_model_output(question, models):
    """Demonstrates a multi-model output using LCEL and Runnables."""

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in five sentence: {question}"
    )

    results = []
    for model_name in models:
        model =init_chat_model(model=model_name,model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
        response = model.invoke(question)
        print(f"Model: {model_name}")
        print(f"Response: {response.content}")
        print("\n")
        results.append(response.content)
    return results

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_basic_streaming()
    demo_multi_chat_model_output("what is AI?", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])