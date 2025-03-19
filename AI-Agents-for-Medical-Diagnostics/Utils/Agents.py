import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")

# Ensure API key is available
if not api_key:
    raise ValueError("Missing API key! Set GEMINI_API_KEY in your .env file.")

# Configure Gemini API
genai.configure(api_key=api_key)

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}  # ✅ Ensure extra_info is always a dictionary
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")  # ✅ Define model here
        self.prompt_template = self.create_prompt_template()

    def create_prompt_template(self):
        templates = {
            "Cardiologist": """
                Act like a cardiologist. Analyze the patient's report for cardiac issues.
                Medical Report: {medical_report}
            """,
            "Psychologist": """
                Act like a psychologist. Analyze the patient's mental health condition.
                Patient's Report: {medical_report}
            """,
            "Pulmonologist": """
                Act like a pulmonologist. Identify potential respiratory issues.
                Patient's Report: {medical_report}
            """,
            "MultidisciplinaryTeam": """
                Act as a team of specialists. Review reports from all three doctors.
                Cardiologist Report: {cardiologist_report}
                Psychologist Report: {psychologist_report}
                Pulmonologist Report: {pulmonologist_report}
            """
        }
        return PromptTemplate.from_template(templates[self.role])

    def run(self):
        prompt_text = self.prompt_template.format(medical_report=self.medical_report, **self.extra_info)

        try:
            response = self.model.generate_content(prompt_text)
            if response and hasattr(response, "text") and response.text.strip():  # ✅ Check if response is valid
                return response.text.strip()
            else:
                return "No diagnosis available."
        except Exception as e:
            print(f"❌ Error occurred while generating response for {self.role}: {e}")
            return "Error generating diagnosis."

# Subclasses for specific roles
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
        self.medical_report = None  # ✅ Explicitly set to None (since it's not required)
