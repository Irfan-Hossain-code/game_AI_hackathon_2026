import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "clash-royal-ai"
LOCATION = "europe-west1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Say hello")
print(response.text)