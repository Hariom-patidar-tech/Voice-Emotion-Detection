from src.speech_to_text import voice_to_text
from src.predict_emotion import predict_emotion

text = voice_to_text()
print(" Text:", text)

if text != "Could not understand audio":
    print(" Logistic Emotion:", predict_emotion(text, "logistic"))
    
else:
    print(" Try speaking again")
