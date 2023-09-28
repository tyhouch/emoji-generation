from transformers import BertTokenizer, BertForSequenceClassification,BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import spacy
from fastapi import FastAPI
# Load pretrained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
# Load pretrained BERT model and tokenizer
model_1 = BertModel.from_pretrained(model_name)
tokenizer_1 = BertTokenizer.from_pretrained(model_name)


# Load your transcriptions from a JSON file (assuming you have 'transcriptions.json')
with open('script.json', 'r') as file:
    dataset = json.load(file)

# Extract text from segments in the dataset
text_list = [segment['text'] for entry in dataset for segment in entry['words']]

# Tokenize the text
tokenized_texts = [tokenizer.tokenize(text) for text in text_list]

# Convert tokenized text to input tensors
input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]

# Add padding to make all sequences the same length
max_length = max(len(tokens) for tokens in input_ids)
input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]

# Convert to tensor
input_ids = torch.tensor(input_ids)

# Forward pass through BERT
with torch.no_grad():
    outputs = model(input_ids)

# Get predicted labels for each input
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Map predicted labels to keywords
# let's assume label 1 indicates the presence of a keyword
keywords = [text for text, label in zip(text_list, predicted_labels) if label == 1]

# Create Emoji Dictionary
emoji_keyword_dict = {
    'happy': '😄',
    'angry': '😡',
    'love': '❤️',
    'funny': '😂',
    'sad': '😢',
    'excited': '😃',
    'confused': '😕',
    'surprised': '😮',
    'cool': '😎',
    'crying': '😭',
    'laughing': '🤣',
    'heart': '❤️',
    'thumbs_up': '👍',
    'thumbs_down': '👎',
    'fire': '🔥',
    'rocket': '🚀',
    'money_bag': '💰',
    'cat': '😺',
    'dog': '🐶',
    '100': '💯',
    'clap': '👏',
    'star': '⭐',
    'sunglasses': '😎',
    'heart_eyes': '😍',
    'alien': '👽',
    'hammer_and_wrench': '🛠️',
    'pizza': '🍕',
    'taco': '🌮',
    'cake': '🍰',
    'coffee': '☕',
    'rainbow': '🌈',
    'umbrella': '☔',
    'moon': '🌙',
    'sun': '☀️',
    'thumbs_up': '👍',
    'thumbs_down': '👎',
    'thinking_face': '🤔',
    'star_struck': '🤩',
    'sweating_smile': '😅',
    'sleeping_face': '😴',
    'party_popper': '🎉',
    'birthday_cake': '🎂',
    'rose': '🌹',
    'bouquet': '💐',
    'fireworks': '🎆',
    'earth_americas': '🌎',
    'earth_africa': '🌍',
    'earth_asia': '🌏',
    'fire_extinguisher': '🧯',
    'hourglass': '⌛',
    'bell': '🔔',
    'ribbon': '🎀',
    'sparkles': '✨',
    'speech_balloon': '💬',
    'left_arrow': '⬅️',
    'right_arrow': '➡️',
    'arrow_up': '⬆️',
    'arrow_down': '⬇️',
    'star_and_crescent': '☪️',
    'peace_symbol': '☮️',
    'radioactive': '☢️',
    'biohazard': '☣️',
    'atom': '⚛️',
    'wheel_of_dharma': '☸️',
    'star_of_david': '✡️',
    'latin_cross': '✝️',
    'orthodox_cross': '☦️',
    'peace_dove': '🕊️',
    'menorah': '🕎',
    'yin_yang': '☯️',
    'wheelchair_symbol': '♿',
    'male_sign': '♂️',
    'female_sign': '♀️',
    'transgender_symbol': '⚧️',
    'hazardous_materials': '🚧',
    'crossed_swords': '⚔️',
    'clinking_glasses': '🥂',
    'tumbler_glass': '🥃',
    'crystal_ball': '🔮',
    'prayer_beads': '📿',
    'gem_stone': '💎',
    'toolbox': '🧰',
    'magnet': '🧲',
    'amphora': '🏺',
    'world_map': '🗺️',
    'mountain': '⛰️',
    'volcano': '🌋',
    'desert': '🏜️',
    'cityscape': '🏙️',
    'ferris_wheel': '🎡',
    'roller_coaster': '🎢',
    'carousel_horse': '🎠',
    'snowflake': '❄️',
    'snowman': '⛄',
    'comet': '☄️',
    'cloud_with_lightning': '🌩️',
    'tornado': '🌪️',
    'fog': '🌫️',
    'water_wave': '🌊',
    # Add more keywords and emojis as needed
}



# Create embeddings for keywords using BERT
def create_embedding(text):
    tokens = tokenizer_1(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_1(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

# Create embeddings for all keywords
keyword_embeddings = {}
for keyword in emoji_keyword_dict.keys():
    keyword_embeddings[keyword] = create_embedding(keyword)

# Calculate similarity between keywords and transcription words
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2).item()

# Calculate similarity for each keyword in the dataset
for entry in dataset:
    for segment in entry['words']:
        text = segment['text']
        text_embedding = create_embedding(text)
        for keyword in keyword_embeddings:
            keyword_embedding = keyword_embeddings[keyword]
            similarity = calculate_similarity(keyword_embedding, text_embedding)
            # Store the similarity score for each keyword in the dataset
            segment.setdefault('keyword_similarities', {})[keyword] = similarity


# Calculate similarity for each keyword in the dataset
for entry in dataset:
    for segment in entry['words']:
        if 'keyword_similarities' in segment:
            keyword_similarities = segment['keyword_similarities']
            best_similarity = 0
            best_emoji = None
            for keyword in keyword_similarities:
                similarity = keyword_similarities[keyword]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_emoji = emoji_keyword_dict.get(keyword)
            if best_emoji:
                segment['recommended_emoji'] = best_emoji
                segment['emoji_score'] = best_similarity
# Print emoji assignments with scores
for entry in dataset:
    for segment in entry['words']:
        if 'recommended_emoji' in segment:
            print(f"Segment: {segment['text']}")
            print(f"Recommended Emoji: {segment['recommended_emoji']}")
            print(f"Emoji Score: {segment['emoji_score']:.4f}")
            print()


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
            best_emoji = emoji_keyword_dict.get(keyword)
    return best_emoji


# Example usage:
input_sentence = "I am feeling happy today"
predicted_emoji = predict_emoji(input_sentence)
print(f"Input Sentence: {input_sentence}")
print(f"Predicted Emoji: {predicted_emoji}")


# Save the keyword embeddings and model to files for later use
torch.save(keyword_embeddings, 'keyword_embeddings.pth')
model_1.save_pretrained('bert_model')

# You can also save the emoji_keyword_dict as a JSON file if needed
with open('emoji_keyword_dict.json', 'w') as json_file:
    json.dump(emoji_keyword_dict, json_file)
