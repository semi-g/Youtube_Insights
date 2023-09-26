"""
This script contains all summarization functionality. This includes:
- Chunking the transcripts into pieces (using different methods)
- Creating embeddings
- Creating relevant prompts
- Interfacing with GPT-3.5-turbo
- Storing the summary    
"""

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMSummarizationCheckerChain
from langchain.docstore.document import Document
from typing import List
import textwrap

load_dotenv()

# Load and split transcript
def load_and_split(file_path: str) -> List[Document]:
    loader = TextLoader(file_path)
    text_doc = loader.load()

    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100) # TODO: first test with small chunk size, to compare quality between methods
    for chunk in splitter.split_documents(text_doc): # If not working -> use split_text
        chunks.append(chunk)

    return chunks

# Create custom prompt template 
# In bullet points, summary not always as good...
def custom_prompt():
    prompt_template = """Write a concise bullet point summary of the following:

    {text}

    CONSCISE SUMMARY IN BULLET POINTS:"""

    BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["text"])
    return BULLET_POINT_PROMPT


# METHOD 1: Map Reduce
def map_reduce(file_path: str, llm = OpenAI(temperature=0)) -> str:
    chunks = load_and_split(file_path)
    map_reduce_chain = load_summarize_chain(llm, 
                                            chain_type="map_reduce", 
                                            verbose=False,)
                                            #map_prompt=BULLET_POINT_PROMPT,
                                            #combine_prompt=BULLET_POINT_PROMPT)
    output_summary = map_reduce_chain.run(chunks)
    wrapped_text = textwrap.fill(output_summary, 
                                width=100, 
                                break_long_words=False,
                                replace_whitespace=False)
    return wrapped_text


# METHOD 2: Stuffing (short videos) -> check for token length
def stuff(file_path: str, llm = OpenAI(temperature=0)) -> str:
    chunks = load_and_split(file_path)
    stuffing_chain = load_summarize_chain(llm,
                                        chain_type="stuff",
                                        verbose=False)
    output_summary = stuffing_chain.run(chunks)
    wrapped_text = textwrap.fill(output_summary,
                                width=100,
                                break_long_words=False,
                                replace_whitespace=False)
    return wrapped_text


# METHOD 3: Refining
def refine(file_path: str, llm = OpenAI(temperature=0)) -> str:
    chunks = load_and_split(file_path)
    prompt_template = """Write a concise but long enough summary to extract all usefull key information of the following:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["text"])

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    REFINE_PROMPT = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    refine_chain = load_summarize_chain(llm, 
                                        chain_type="refine", 
                                        verbose=False,
                                        question_prompt=PROMPT,
                                        refine_prompt=REFINE_PROMPT)

    output_summary = refine_chain({"input_documents": chunks}, return_only_outputs=True)
    wrapped_text = textwrap.fill(output_summary['output_text'],
                                width=100, 
                                break_long_words=False,
                                replace_whitespace=False)
    return wrapped_text


# Fact checking using langchain
def extract_facts(file_path: str, llm = OpenAI(temperature=0)) -> str:
    chunks = load_and_split(file_path)
    fact_extraction_prompt = PromptTemplate(
        input_variables=["text_input"],
        template="Extract the key facts out of this text. Don't include opinions. \
        Give each fact a number and keep them short sentences. :\n\n {text_input}"
    )
    fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
    facts = fact_extraction_chain.run(chunks)
    wrapped_text = textwrap.fill(facts,
                                width=100,
                                break_long_words=False,
                                replace_whitespace=False)
    return wrapped_text


# Problem with this method: Where does the fact checking information come from? If from gpt-3.5's knowledge base,
# how to be sure it is not hallucinating these facts? 
# e.g. 
# - Game industry traditionally works the hardest and is paid the least: False - While the game industry is known for long hours and hard work, it is not necessarily the lowest paid industry.
# - Closer to a job's passion, the more likely to be underpaid: False - While it is possible to be underpaid in a job related to one's passion, it is not necessarily the case.
# While in reality these statements above may be false, they were said in the video -> so they are true according to the video that is being summarized
# and therefore should be included in the summary.
def check_facts(file_path: str, llm = OpenAI(temperature=0)) -> str:
    chunks = load_and_split(file_path)
    fact_checker_chain = LLMSummarizationCheckerChain(llm=llm,
                                                verbose=True, # To see the process (prints out steps)
                                                max_checks=2
                                                )
    final_summary = fact_checker_chain.run(chunks)
    final_summary


# Fact Checking (post-processing) using named entity/ quantity recognition and verification -> this one should be used of maximal value
# TODO

