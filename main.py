from fastapi import FastAPI
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
app = FastAPI()

# Load pretrained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load pretrained BERT model and keyword embeddings
model_1 = BertModel.from_pretrained('bert_model')
keyword_embeddings = torch.load('keyword_embeddings.pth')


# Load emoji_keyword_dict if needed
with open('emoji_keyword_dict.json', 'r') as json_file:
     emoji_keyword_dict = json.load(json_file)


# Create embeddings for input text
def create_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_1(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

# Calculate similarity between keywords and input text
def predict_emoji(input_text):
    text_embedding = create_embedding(input_text)
    best_similarity = 0
    best_emoji = None
    for keyword in keyword_embeddings:
        keyword_embedding = keyword_embeddings[keyword]
        similarity = cosine_similarity(keyword_embedding, text_embedding).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_emoji = emoji_keyword_dict.get(keyword)  # Use the emoji_keyword_dict here
    return best_emoji, best_similarity

# FastAPI endpoint for emoji prediction
@app.post("/predict-emoji/")
async def predict_emoji_endpoint(input_text: str):
    predicted_emoji, similarity_score = predict_emoji(input_text)
    return {"input_text": input_text, "predicted_emoji": predicted_emoji, "similarity_score": similarity_score}
