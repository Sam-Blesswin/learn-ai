import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import FewShotChatMessagePromptTemplate
load_dotenv()


def demo_prompt():
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in five sentence: {question}"
    )
    messages = prompt.format_messages(question="What is the capital of France?")
    print(messages)

def demo_prompts():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("user", "Translate the following sentence: {sentence}")
    ])
    messages = prompt.format_messages(input_language="English", output_language="French", sentence="What is the capital of France?")
    print(messages)

def demo_model_with_prompt(input_language, output_language, sentence):
    """Demonstrates a basic chain using LCEL and Runnables."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("user", "Translate the following sentence: {sentence}")
    ])
    messages = prompt.format_messages(input_language=input_language, output_language=output_language, sentence=sentence)

    model =init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])

    response = model.invoke(messages)
    print(f"Response: {response.content}")

def demo_few_shot_prompt(country):
    examples = [
        {
            "input": "What is the capital of France?",
            "output": "Paris"
        },
        {
            "input": "What is the capital of Germany?",
            "output": "Berlin"
        }
    ]
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ]),
    )
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Give the capital of the following country"),
        few_shot_prompt,
        ("user", "{country}"),
    ])
    messages = final_prompt.format_messages(country=country)
    print(messages)
    
    model =init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    response = model.invoke(messages)
    print(f"Response: {response.content}")

if __name__ == "__main__":
    #demo_model_with_prompt(input_language="English", output_language="French", sentence="What is the capital of France?")
    demo_few_shot_prompt(country="India")