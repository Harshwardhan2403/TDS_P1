from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import openai
from tasksA import A1, A2  # Import actual functions from tasksA.py
from tasksB import B1, B2  # Import actual functions from tasksB.py

app = FastAPI()

openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjIwMDA2OTBAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ilj3P52UqKwCM16fpkCGnngK0HhZnKMZAprMx7FEExM"  # Replace with your OpenAI API Key

class RequestModel(BaseModel):
    user_request: str

def get_completions(user_request):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_request}]
    )
    return response["choices"][0]["message"]["content"].strip()

@app.post("/execute-task/")
async def execute_task(request: RequestModel):
    result = get_completions(request.user_request)
    try:
        task_name, arguments = result.split("(")
        arguments = arguments.rstrip(")")
        function_to_call = globals().get(task_name)
        if function_to_call:
            return {"result": function_to_call(**json.loads(arguments))}
        else:
            raise HTTPException(status_code=400, detail="Invalid task name")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
