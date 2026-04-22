"""Register a prompt as v1, then (in Step 2) fix it and register v2."""
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


USER_QUESTION = "My flight was delayed 2 hours. Am I entitled to compensation?"


@mlflow.trace
def ask_air_assist(prompt_version: int, question: str) -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/airAssist-system-prompt/{prompt_version}")
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt.format(user_input=question)}],
    )
    return response.choices[0].message.content.strip()


# --- Step 1 ----------------------------------------------------------------

AIR_ASSIST_PROMPT_V1 = """You are AirAssist. Follow this rule exactly:
  - If the flight delay is under 3 hours, reply: "Yes, you are entitled to €600."
  - Otherwise, reply: "Please contact your airline's customer service."
Passenger question: {{ user_input }}"""

v1 = mlflow.genai.register_prompt(
    name="airAssist-system-prompt",
    template=AIR_ASSIST_PROMPT_V1,
    commit_message="Initial version",
)
ask_air_assist(v1.version, USER_QUESTION)

print(
    f"\nStep 1 done. airAssist-system-prompt registered as v{v1.version}.\n"
    "In the MLflow UI, open the prompt registry and find the trace for this 'ask_air_assist' call.\n"
    "Read v1's answer. If a customer got this reply, what would it commit the airline to?\n"
)


# --- Step 2 ----------------------------------------------------------------
# TODO — AIR_ASSIST_PROMPT_V1 has a bug that would authorise the wrong behaviour.
# Read it, find the bug, copy the template below, fix it, and register as v2.
#
# AIR_ASSIST_PROMPT_V2 = """..."""
#
# v2 = mlflow.genai.register_prompt(
#     name="airAssist-system-prompt",
#     template=AIR_ASSIST_PROMPT_V2,
#     commit_message="<describe what you changed and why>",
# )
# ask_air_assist(v2.version, USER_QUESTION)
#
# print(
#     f"\nStep 2 done. airAssist-system-prompt registered as v{v2.version}.\n"
#     "In the MLflow UI, compare v1 against v2 on the prompt registry page.\n"
#     "What changed, and what would rollback to v1 take if a customer complained about v2 tomorrow?\n"
# )
