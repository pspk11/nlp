import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class EmailAutocompleteSystem:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

    def generate_suggestions(self, user_input, context):
        input_text = f"{context} {user_input}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=50, num_return_sequences=1,no_repeat_ngram_size=2)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        suggestions = generated_text.split()[len(user_input.split()):]
        return suggestions

if __name__ == "__main__":
    autocomplete_system = EmailAutocompleteSystem()
    email_context = "Subject: Discussing Project Proposal\nHi [Recipient],"
    while True:
        user_input = input("Enter your sentence (type 'exit' to end): ")
        if user_input.lower() == 'exit':
            break
        suggestions = autocomplete_system.generate_suggestions(user_input, email_context)
        if suggestions:
            print("Autocomplete Suggestions:", suggestions)
        else:
            print("No suggestions available.")
