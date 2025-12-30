import speech_recognition as sr

def voice_to_text(lenguage="en-IN"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙 Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Could not understand audio"
