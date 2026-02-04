import os
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image


# ==============================
# PATH HANDLING (VERY IMPORTANT)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "image_caption_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")


# ==============================
# LOAD MODEL & TOKENIZER
# ==============================

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# ==============================
# LOAD CNN FEATURE EXTRACTOR
# ==============================

efficientnet = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


# ⚠️ MUST MATCH TRAINING VALUE
max_caption_len = 35


# ==============================
# IMAGE PREPROCESSING
# ==============================

def preprocess_image(img_path):
    """
    Load and preprocess image for EfficientNet
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# ==============================
# CAPTION GENERATION
# ==============================

def generate_caption(photo_feature):
    """
    Generate caption from extracted image feature
    """
    in_text = "startseq"

    for _ in range(max_caption_len):
        # Convert words to sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # ✅ FIX: pad to EXACT training length
        sequence = pad_sequences(
            [sequence],
            maxlen=max_caption_len,
            padding="post"
        )

        # Predict next word
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    # Clean output before returning
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


