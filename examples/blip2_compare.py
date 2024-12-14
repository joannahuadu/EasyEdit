# Load model directly
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

print("model")