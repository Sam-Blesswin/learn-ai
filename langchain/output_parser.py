import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

def demo_basic_str_output_parser():
    """Demonstrates a basic streaming chain using LCEL and Runnables."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in five sentence: {question}"
    )
    model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"question": "What is LangChain?"})
    print(result)

def demo_basic_json_output_parser():
    """Demonstrates a basic JSON output parser using LCEL and Runnables."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Return the answer in JSON format. The JSON should have the following fields: question:  {question},answer"
    )
    model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    parser = JsonOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"question": "What is LangChain?"})
    print(result)

def demo_pydantic_output_parser():
    """Demonstrates a basic JSON output parser with schema using LCEL and Runnables."""
    model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])

    class ResponseSchema(BaseModel):
        question: str
        answer: str
    
    parser = PydanticOutputParser(pydantic_object=ResponseSchema)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Return the answer in JSON format. The JSON should have the following fields: question:  {question},answer"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    result = chain.invoke({"question": "What is LangChain?"})
    print(result)

def demo_structured_output_parser():
    """Demonstrates a basic structured output parser using LCEL and Runnables."""
    class MoviewReview(BaseModel):
        movie_name: str = Field(description="The name of the movie")
        summary: str = Field(description="A short summary of the movie")
        rating: int = Field(description="The rating of the movie")

    model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq",temperature=0,api_key=os.environ["GROQ_API_KEY"])
    structured_model = model.with_structured_output(MoviewReview)
    result = structured_model.invoke("batman movie is a great movie")
    print(result)

if __name__ == "__main__":
    # demo_basic_str_output_parser()
    # demo_basic_json_output_parser()
    # demo_pydantic_output_parser()
    demo_structured_output_parser()