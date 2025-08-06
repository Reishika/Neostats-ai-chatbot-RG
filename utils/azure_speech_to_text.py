import azure.cognitiveservices.speech as speechsdk

def transcribe_speech_from_mic(subscription_key: str, region: str) -> str:
    # Set up the speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    # Set the language, optional
    speech_config.speech_recognition_language = "en-US"

    # Create a recognizer with the default microphone
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Speak into your microphone...")

    # Start speech recognition
    result = speech_recognizer.recognize_once_async().get()

    # Check result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:", result.text)
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled:", cancellation_details.reason)
        return ""

    return ""
