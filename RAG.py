from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings# free stuff
loader = CSVLoader(
    file_path = '/content/course_data.csv',
    source_column = "Class_name", #this is what we are tagging by
    content_columns = [ #data
        'Class_name',
        "Teacher",
        "Difficulty",
        "homework_load",
        "teacher_quality",
        'interest_level' # add classes if you'd like
        # we should add subjects
    ]
)
data = loader.load() #splits the data up into smaller peices
#I added a text splitter if text data is added
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,  # charectors per chunk
    chunk_overlap=50,  # chunk overlap idk what to change this number to appaerntly 50 is good
    add_start_index=True,  # idk what this does
)
all_splits = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", show_progress=False  ) #i have no idea what this line does
vector_store = FAISS.from_documents(all_splits, embeddings)
from langchain.tools import tool
def df_to_string(df): # i stole this from sid
    text = ""
    for class_name in df["Class_name"].unique():
        text += f"\n Class: {class_name}\n"
        class_df = df[df["Class_name"] == class_name]
        for teacher in class_df["Teacher"].unique():
            teacher_df = class_df[class_df["Teacher"] == teacher]
            avg_difficulty = teacher_df["Difficulty"].mean()
            avg_homework = teacher_df["homework_load"].mean()
            avg_quality = teacher_df["teacher_quality"].mean()
            avg_interest = teacher_df["interest_level"].mean()
            text += f"\n Teacher: {teacher}\n Avg difficulty: {avg_difficulty}\n Avg Homework Load: {avg_homework}\n Avg Teacher Quality: {avg_quality}\n Average Interest: {avg_interest}"
    return text
numerical_summary=df_to_string(pd.read_csv('/content/course_data.csv'))
# is an ai just to sort through the data

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=20) #change the k value when we get more data
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    full_context = f"""
  SUMMARY STATISTICS:
  {numerical_summary}

  RELEVANT ROWS:
  {serialized}
"""
    return full_context, retrieved_docs
print(retrieve_context.invoke("Biology is my favorite class ive ever taken")) #test

#Use google-genai text embedding model
'''
from google import genai
import os

client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

client = client.models.embed_content(
    models = "gemini-embedding-001"
    content = content
)
'''
