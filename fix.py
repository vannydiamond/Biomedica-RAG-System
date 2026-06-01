import cohere
import os

co = cohere.Client(os.getenv("l3tgfnqTsk4Ys8h8uIY9N2uhiIt3T2rie97SGPFSz"))  # Replace with your actual API key

models = co.models.list()  # correct API call
print(models)