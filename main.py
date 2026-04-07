from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

course_data = pd.read_csv("course_data/Course_Data_Examples.csv")

def df_to_string(df):
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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Question):
    prompt = f"""You are a helpful course advisor. Answer in 1 sentence.
Use ONLY the course info below. If the answer is not in the data, say 'Unknown'.
Numbers are on a scale of 1-5.

Course data:
{course_data}

Student question: {body.question}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response.choices[0].message.content}








