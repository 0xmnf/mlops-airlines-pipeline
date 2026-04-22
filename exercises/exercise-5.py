"""Enable autolog, then send three different-size questions through the LLM."""
import os
import warnings
import logging

import mlflow
import mlflow.openai
from dotenv import load_dotenv
from openai import AzureOpenAI

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


AZURE_ENDPOINT    = "https://mlops-brisa-trainning-resource.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT  = "gpt-5.4-nano"

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

mlflow.set_experiment("<pick an experiment name>")
mlflow.openai.autolog()


QUESTIONS = [
    "Answer with one word only: can flights be delayed?",
    "In one sentence, name a common winter delay cause.",
    "Write a detailed essay on the top five causes of flight delays in winter, with two examples each and common mitigations.",
]


for question in QUESTIONS:
    client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are AirAssist. Answer flight delay questions only."},
            {"role": "user",   "content": question},
        ],
    )


print(
    "Done.\n"
    "In the MLflow UI, open this experiment and find the traces from this run.\n"
    "How do latency and token counts differ across the three questions? What would that mean at 10,000 calls a day?"
)
