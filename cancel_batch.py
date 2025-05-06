import openai
from openai import OpenAI

client = OpenAI()


client.batches.cancel("batch_")