# import necessaary libraries
import os
import pandas as pd
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, RetrievalQA
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema.runnable import RunnableLambda


# loading the environment variables

load_dotenv()

os.environ.get("HUGGING_FACE_API")
os.environ.get("GROQ_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ.get("LANGSMITH_API_KEY")


# loading the necessary documents
def load_csv(path):
    df = pd.read_csv(path)
    return df


def load_pdf_and_embed(path):
    doc = PyPDFLoader(path)
    docs = doc.load()
    text = "\n".join([doc.page_content for doc in docs])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
    splitter = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    client = QdrantClient(":memory:")
    client.create_collection(
    collection_name="router",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)


    vector_store = QdrantVectorStore(
        client=client, collection_name="router", embedding=embeddings,
    )

    vector = vector_store.add_texts(texts=splitter) 
    return vector_store


# initialising the prompts required to function the LLM

# Prompt 1: General Router Prompt
general_router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a skilled customer service manager and information provider and router. Your primary responsibility is to classify the user query into one of the predefined task types.

### Task Types:
- **FAQ** → For general questions (e.g., "How to track orders?" or "How to contact support?").
- **Order Lookup** → When the user provides an order ID (e.g., "Track my order ID: 1001").
- **Product Info** → When the user inquires about a specific product (e.g., "Track my laptop").

### Examples:
- **Question**: "How to track orders?"
Output
  **Task Type**: FAQ

- **Question**: "Track my order ID: 1001"
Output
  **Task Type**: Order Lookup

- **Question**: "Track my laptop"
Output
  **Task Type**: Product Info


Now, classify the following user query strictly based on the above categories.

### User Query:
{query}

### Output:
**Task Type:** (Only return one of the following: FAQ / Order Lookup / Product Info)
""",
)

# Prompt 2: FAQ Prompt for Fetching Answers
faq_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Your primary goal is to accurately answer user questions by utilizing information from the company database. Use Retrieval-Augmented Generation (RAG) to fetch and present the most relevant data.

Question: {query}

In your responses, incorporate the following relevant data points that got retrieved and answer with precise context:
""",
)

# Prompt 3: Order Lookup for Order & Product Retrieval
order_lookup_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an expert in analyzing customer orders and providing detailed and accurate information. Your primary role is to utilize the provided tools to efficiently look up order numbers, retrieve relevant details about the orders, and address any questions or concerns the user may have. Always aim to deliver clear, concise, and helpful responses, ensuring the user's needs are fully met.

Lookup order numbers and product IDs using the CSV:
- One is orders variable containing order details.
- The other is products variable containing product information.

Orders always contain an array of productIdsOrdered. Use these IDs to look up the specific products from the product lookup tool and aggregate the product information with the order to provide a clear summary.

If the order does not exist, simply tell the user to try again as the ID wasn't found.

Only return information about orders, do not return anything else.

User Query: {query}
""",
)

# initialising the LLM
llm = ChatGroq(model_name="deepseek-r1-distill-qwen-32b", temperature=0.3)

# calling the document loading functions
order = load_csv(r"D:\OneDrive\Desktop\karthik\Projects\LLM_SCC\orders.csv")
product = load_csv(r"D:\OneDrive\Desktop\karthik\Projects\LLM_SCC\products.csv")
faq = load_pdf_and_embed(
    r"D:\OneDrive\Desktop\karthik\Projects\LLM_SCC\Company_FAQ.pdf"
)


# functions to identify the task based on the user query
def identify_task(query):
    router_chain = general_router_prompt | llm | RunnableLambda(lambda x: x)
    task_type = router_chain.invoke({'query':query})
    return task_type


# retrieve answer from faq
def retrieve_from_faq(query, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    results = qa_chain.invoke({"query":query})

    print(results)
    result_text = results['result']
    response = result_text.split("\n\n")
    final_answer = response[-1]
    return final_answer


# fetch order and product details
def fetch_order_details(order_id, order_df, product_df):
    order_info = order_df[order_df["orderNumber"] == int(order_id)]
    if not order_info.empty:
        order_info = order_info.to_dict(orient="records")[0]

        product_ids = order_info.get("productIdsOrdered", "").split(
            ","
        )  # Assuming CSV stores product IDs as comma-separated
        product_details = product_df[
            product_df["productId"].astype(str).isin(product_ids)
        ]
        print("product_details",product_details)

        order_summary = {
            "order_id": order_info["orderNumber"],
            "status": order_info["status"],
            "delivery_date": order_info["date"],
            "products": product_details.to_dict(orient="records"),
        }
        return order_summary

    return "Order ID not found. Please try again."


# parent function
def chatbot_rag(query, order_df, product_df, vectorstore):
    task_type = identify_task(query)
    def extract_task_type(response_text):
        match = re.search(r"\*\*Task Type:\*\* (FAQ|Order Lookup|Product Info)", response_text)
        return match.group(0) if match else "Task Type not found"
    # Extract Task Type
    task_type = extract_task_type(str(task_type))
    print(task_type)
    

    if "order lookup" in task_type.lower():
        order_lookup_chain = order_lookup_prompt| llm | RunnableLambda(lambda x: x)
        response = order_lookup_chain.invoke({"query":query})

        if isinstance(response, dict):  
            response_text = response.get("text", "")
    
        else:
            response_text = str(response)
            
        # Regex pattern to extract only the message after </think>
        match = re.search(r"</think>\n\n \s*\n*(.*)", response_text, re.DOTALL)

        if match:
            message = match.group(1).strip()
            print(message)
        else:
            print("No message found.")
    
        """if order_id:
            return fetch_order_details(order_id, order_df, product_df)
        else:
            return 'Order ID not found. Please try again.'"""
    

    elif "faq" in task_type.lower():
        return retrieve_from_faq(query, vectorstore)

    elif "product info" in task_type.lower():
        product_match = product_df[
            product_df["product_name"].str.contains(query, case=False, na=False)
        ]
        if not product_match.empty:
            return product_match.to_dict(orient="records")[0]
        else:
            return "Product information not found."

    else:
        return "I'm not sure how to handle your request. Can you clarify?"


query = input("Enter your query: ")
response = chatbot_rag(query, order, product, faq)
print(response)
