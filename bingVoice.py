
import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
import warnings
import sys
from playsound import playsound

# Initialize the OpenAI API
openai.api_key = "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"

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "hola"
GPT_WAKE_WORD = "gpt"
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    else:
        return None

def speak(text):
    # If Mac OS use system messages for TTS
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$: ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f"say '{clean_text}'")
    # Use pyttsx3 for other operating sytstems
    else:
        import pyttsx3
        engine = pyttsx3.init() 
        # Get the current speech rate
        rate = engine.getProperty('rate')
        # Decrease speech rate by 50 words per minute (Change as desired)
        engine.setProperty('rate', rate-50) 

        engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ES-MX_SABINA_11.0')#voice local
        engine.say(text)
        engine.runAndWait()

def speak_azure(speak):
    import azure.cognitiveservices.speech as speechsdk

    # Creates an instance of a speech config with specified subscription key and service region.
    speech_key = "c55c6d9bb32245c69d81af8285c92257"
    service_region = "eastus"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "es-MX-JorgeNeural"

    text = speak

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = speech_synthesizer.speak_text_async(text).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def speak_google(speak_text):
    from gtts import gTTS
    tts = gTTS(text=speak_text, lang='es')
    tts.save("audio_google.mp3")
    from pydub import AudioSegment
    from pydub.playback import play
    audio = AudioSegment.from_file("audio_google.mp3")
    play(audio)

def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)

async def main():
    while True:
        with sr.Microphone() as source:#micro disponible
            recognizer.adjust_for_ambient_noise(source)
            print('\n"Bing: En el momento que me necesites, di "¡Hola Bing!" para activarme. \n')
            #speak('En el momento que me necesites, di "¡Hola Bing!" para activarme.')
            speak_google('En el momento que me necesites, di "¡Hola Bing!" para activarme.')
            #speak_azure('En el momento que me necesites, di "¡Hola Bing!" para activarme.')
            while True:
                audio = recognizer.listen(source, timeout=5)
                try:
                    with open("audio_key.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    # Use the preloaded tiny_model
                    model = whisper.load_model("base")
                    result = model.transcribe("audio_key.wav")
                    phrase = result["text"]
                    print(f"Tú: {phrase}")

                    wake_word = get_wake_word(phrase)                    
                    if wake_word is not None:
                        # Play wake word detected notification sound (faster than TTS)                        
                        print("Bing: Palabra de activación detectada. ¿En qué te puedo ayudar?. \n")
                        #speak("Palabra de activación detectada. ¿En qué te puedo ayudar?.")
                        speak_google("Palabra de activación detectada. ¿En qué te puedo ayudar?.")       
                        #speak_azure("Palabra de activación detectada. ¿En qué te puedo ayudar?.")
                        break
                    else:
                        print("Bing: No se encontró ninguna palabra de activación. Seguiré escuchando...")
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue
            while True:
                try:
                    playsound('wake_detected.mp3')
                    # Record prompt
                    audio = recognizer.listen(source)
                    with open("audio_prompt.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    model = whisper.load_model("base")
                    result = model.transcribe("audio_prompt.wav")
                    user_input = result["text"]                    
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue

                if len(user_input.strip()) == 0:
                        #playsound("emptyPrompt.mp3")
                        #speak("Empty prompt. Please speak again.") 
                        print("Bing: Mensaje vacío. Por favor, hable de nuevo.")
                        #speak("Mensaje vacío. Por favor, hable de nuevo.") 
                        speak_google("Mensaje vacío. Por favor, hable de nuevo.") # voz google                      
                        #speak_azure("Mensaje vacío. Por favor, hable de nuevo.") # voz microsoft                           
                        continue
                else:
                    if 'gracias' in user_input.lower().strip():
                        # Play wake word detected notification sound (faster than TTS)                        
                        print("Bing: De nada, ¡Adios!. \n")
                        #speak("De nada, ¡Adios!.")
                        speak_google("De nada, ¡Adios!.")
                        #speak_azure("De nada, ¡Adios!.") # voz microsoft       
                        break
              
                if wake_word == BING_WAKE_WORD:  
                    print(f"\nSending: {user_input}")          
                    import json
                    # Lee el contenido del archivo JSON
                    with open('cookies.json') as f:
                        cookies_content = json.load(f)
                    # Pasa el contenido como argumento al constructor de la clase Chatbot
                    bot = Chatbot(cookies=cookies_content)

                    response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)
                    # Select only the bot response from the response dictionary
                    for message in response["item"]["messages"]:
                        if message["author"] == "bot":
                            bot_response = message["text"]
                    # Remove [^#^] citations in response
                    bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
                    print("Response:", bot_response)
                    #speak(bot_response)
                    speak_google(bot_response)
                    #speak_azure(bot_response)
                    await bot.close()
                # gpt
                else:
                    print(f"Sending: {user_input}")
                    # Send prompt to GPT-3.5-turbo API
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content":
                            "You are a helpful assistant."},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=0.5,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n=1,
                        stop=["\nUser:"],
                    )

                    bot_response = response["choices"][0]["message"]["content"]
                    print("Response:", bot_response)
                    #speak(bot_response)
                    speak_google(bot_response)
                    #speak_azure(bot_response)
                    await bot.close()
        
if __name__ == "__main__":
    asyncio.run(main())

