# Import the updated agent classes
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import os
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API Key
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ API Key is missing! Make sure GEMINI_API_KEY is set in the .env file.")

print(f"✅ API Key Loaded: {api_key[:6]}****")  # Hide part of the key for security
genai.configure(api_key=api_key)

# Initialize model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Read the medical report
medical_report_path = "AI-Agents-for-Medical-Diagnostics/Medical Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt"
if not os.path.exists(medical_report_path):
    raise FileNotFoundError(f"❌ Medical report file not found at: {medical_report_path}")

with open(medical_report_path, "r", encoding="utf-8") as file:
    medical_report = file.read()

# Initialize individual agents
agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# Function to run each agent and get their response
def get_response(agent_name, agent):
    try:
        response = agent.run()
        if response:
            return agent_name, response
        else:
            print(f"⚠️ {agent_name} did not return a valid response!")
            return agent_name, "No response available."
    except Exception as e:
        print(f"❌ Error while running {agent_name}: {e}")
        return agent_name, "Error in response."

# Run the agents concurrently and collect responses
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

# Debugging: Print collected responses
print("\n🔍 Collected Responses from Specialists:")
for key, value in responses.items():
    print(f"🩺 {key}: {value[:100]}...")  # Print first 100 characters of each response

# Ensure responses exist before running MultidisciplinaryTeam
if any(responses[key] == "No response available." for key in responses):
    raise RuntimeError("❌ Some agents did not return responses! Cannot generate final diagnosis.")

# Initialize and run the MultidisciplinaryTeam agent
team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"],
    pulmonologist_report=responses["Pulmonologist"]
)

# Get the final diagnosis
final_diagnosis = team_agent.run()
if not final_diagnosis or final_diagnosis.strip() == "":
    final_diagnosis = "No diagnosis available."

# Debugging Step: Print before saving
print("\n📄 Final Diagnosis Generated:")
print(final_diagnosis[:500])  # Print first 500 characters

# Ensure results directory exists
txt_output_path = "results/final_diagnosis.txt"
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

# Write to file
try:
    with open(txt_output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"### Final Diagnosis:\n\n{final_diagnosis}")

    print(f"\n✅ Final diagnosis has been saved to {txt_output_path}")
except Exception as e:
    print(f"❌ Error while saving the final diagnosis: {e}")
