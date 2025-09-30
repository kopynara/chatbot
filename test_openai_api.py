from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.models.list()
print([m.id for m in resp.data if "gpt" in m.id])
