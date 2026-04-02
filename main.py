import pandas as pd
import ollama

course_data = pd.read_csv("course_data/Course_Data_Examples.csv")
course_data = course_data.dropna()
course_info = course_data.to_string(index=False)

client = ollama.Client()
model = "llama2"
student_prompt = input("What course questions do you have:")
prompt = f"You are a helpful course advisor Answer the student prompt only. Answer in 1 sentence. Answer questions **using ONLY the course info**.Do NOT make up any information. If the answer is not present in the data, say 'Unknown'.Here is the class data: {course_info}. Answer questions using only the course info. Numbers are on scale of 1-5. Here is student prompt {student_prompt}"

response = client.generate(model=model, prompt=prompt)
print(response["response"])
