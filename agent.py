import os
import shutil
import time

# Folder structure
folder_name = '/mnt/data/build-your-first-agent-azure-gpt5'
os.makedirs(folder_name, exist_ok=True)

# agent.py (Azure OpenAI GPT-5 Mini version)
agent_code = '''"""
Agentic AI Workshop Example - GPT-5 Mini via Azure/OpenAI
This agent:
1. Receives a high-level task
2. Uses GPT-5 Mini to plan steps
3. Simulates execution of each step
"""

import os
import time
from openai import OpenAI

# ===============================
# Configuration: Azure/OpenAI
# ===============================
# Participants can provide their API key
AZURE_API_KEY = os.getenv("AZURE_API_KEY") or "YOUR_API_KEY_HERE"
AZURE_API_BASE = os.getenv("AZURE_API_BASE") or "https://YOUR_RESOURCE_NAME.openai.azure.com/"

client = OpenAI(
    api_key=AZURE_API_KEY,
    api_base=AZURE_API_BASE,
    api_type="azure",
    api_version="2023-07-01-preview"
)

# ===============================
# Agentic functions
# ===============================
def get_plan(task: str):
    prompt = f"Task: {task}\\nBreak this task into 3-5 actionable steps for an AI agent."
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content
    steps = text.split('\\n')
    steps = [step.strip().lstrip('0123456789.- ') for step in steps if step.strip()]
    return steps

def perform_step(step: str):
    print(f"🔹 Performing: {step}...")
    time.sleep(1)
    print(f"✅ Completed: {step}\\n")
    return f"Step '{step}' done."

# ===============================
# Main loop
# ===============================
def main():
    print("🤖 Welcome to your Azure GPT-5 Mini Agentic AI agent!")
    print("Give me a task, and I will plan and execute it for you.")
    print("Type 'exit' to quit.\\n")

    while True:
        task = input("You: ")
        if task.lower() == 'exit':
            print("Goodbye! 👋")
            break

        print("\\n📝 Planning steps...")
        plan_steps = get_plan(task)
        for i, step in enumerate(plan_steps, 1):
            print(f"{i}. {step}")

        print("\\n🚀 Executing steps...")
        for step in plan_steps:
            perform_step(step)

        print("🎉 All steps completed!\\n")

if __name__ == "__main__":
    main()
'''

with open(os.path.join(folder_name, 'agent.py'), 'w') as f:
    f.write(agent_code)

