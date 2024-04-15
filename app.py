import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")

def parse_pdf(file_name):
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2,
    )

    documents = parser.load_data(file_name)
    return documents

documents = parse_pdf("section-1-general-introduction-v4.pdf")

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo"), num_workers=8)

nodes = node_parser.get_nodes_from_documents(documents=[documents[0]])

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=base_nodes+objects)

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[reranker],
    verbose=True
)

query = "How many hectares does residential zone have?"
response = recursive_query_engine.query(query)

print(response)