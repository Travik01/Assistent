from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from vosk import Model, KaldiRecognizer
from googlesearch import search
from pyowm import OWM
from termcolor import colored
from dotenv import load_dotenv
import speech_recognition
import googletrans
import pyttsx3
import wikipediaapi
import random
import webbrowser
import traceback
import json
import wave
import os


class Translation:
    with open("translations1.json", "r", encoding="UTF-8") as file:
        translations = json.load(file)

    def get(self, text: str):

        if text in self.translations:
            return self.translations[text][assistant.speech_language]
        else:
            print(colored("Not translated phrase: {}".format(text), "red"))
            return text


class OwnerPerson:
    name = "Maria"
    home_city = "Tyumen"
    native_language = "ru"
    target_language = "en"


class VoiceAssistant:
    name = "Djon"
    sex = "Men"
    speech_language = "ru"
    recognition_language = "en"


def setup_assistant_voice():

    voices = ttsEngine.getProperty("voices")
    for voice in voices:
        if voice.name == 'Aleksandr':
            ttsEngine.setProperty('voice', voice.id)

    if assistant.speech_language == "en":
        assistant.recognition_language = "en-US"
        if assistant.sex == "female":
            ttsEngine.setProperty("voice", voices[1].id)
        else:
            ttsEngine.setProperty("voice", voices[0].id)
    else:
        assistant.recognition_language = "ru-RU"
        ttsEngine.setProperty("voice", voices[0].id)


def record_and_recognize_audio(*args: tuple):
    with microphone:
        recognized_data = ""

        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 10)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            play_voice_assistant_speech(translator.get("Can you check if your microphone is on, please?"))
            traceback.print_exc()
            return
        try:

            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language=assistant.recognition_language).lower()

        except speech_recognition.UnknownValueError:
            pass  # play_voice_assistant_speech("What did you say again?")

        except speech_recognition.RequestError:
            print(colored("Trying to use offline recognition...", "cyan"))
            recognized_data = use_offline_recognition()

        return recognized_data


def use_offline_recognition():

    recognized_data = ""
    try:
        if not os.path.exists("models/vosk-model-small-" + assistant.speech_language + "-0.4"):
            print(colored("Please download the model from:\n"
                          "https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.",
                          "red"))
            exit(1)

        wave_audio_file = wave.open("microphone-results.wav", "rb")
        model = Model("models/vosk-model-small-" + assistant.speech_language + "-0.4")
        offline_recognizer = KaldiRecognizer(model, wave_audio_file.getframerate())

        data = wave_audio_file.readframes(wave_audio_file.getnframes())
        if len(data) > 0:
            if offline_recognizer.AcceptWaveform(data):
                recognized_data = offline_recognizer.Result()

                recognized_data = json.loads(recognized_data)
                recognized_data = recognized_data["text"]
    except:
        traceback.print_exc()
        print(colored("Sorry, speech service is unavailable. Try again later", "red"))

    return recognized_data


def play_voice_assistant_speech(text_to_speech):

    ttsEngine.say(str(text_to_speech))
    ttsEngine.runAndWait()


def play_failure_phrase(*args: tuple):

    failure_phrases = [
        translator.get("Can you repeat, please?"),
        translator.get("What did you say again?")
    ]
    play_voice_assistant_speech(failure_phrases[random.randint(0, len(failure_phrases) - 1)])


def play_greetings(*args: tuple):

    greetings = [
        translator.get("Hello, {}! How can I help you today?").format(person.name),
        translator.get("Good day to you {}! How can I help you today?").format(person.name)
    ]
    play_voice_assistant_speech(greetings[random.randint(0, len(greetings) - 1)])


def play_farewell_and_quit(*args: tuple):
    farewells = [
        translator.get("Goodbye, {}! Have a nice day!").format(person.name),
        translator.get("See you soon, {}!").format(person.name)
    ]
    play_voice_assistant_speech(farewells[random.randint(0, len(farewells) - 1)])
    ttsEngine.stop()
    quit()


def search_for_term_on_google(*args: tuple):
    if not args[0]: return
    search_term = " ".join(args[0])
    url = "https://google.com/search?q=" + search_term
    webbrowser.get().open(url)

    search_results = []
    try:
        for _ in search(search_term,
                        tld="com",
                        lang=assistant.speech_language,
                        num=1,
                        start=0,
                        stop=1,
                        pause=1.0,
                        ):
            search_results.append(_)
            webbrowser.get().open(_)

    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    print(search_results)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on google").format(search_term))


def search_for_video_on_youtube(*args: tuple):

    if not args[0]: return
    search_term = " ".join(args[0])
    url = "https://www.youtube.com/results?search_query=" + search_term
    webbrowser.get().open(url)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on youtube").format(search_term))


def search_for_definition_on_wikipedia(*args: tuple):

    if not args[0]: return

    search_term = " ".join(args[0])

    wiki = wikipediaapi.Wikipedia(assistant.speech_language)

    wiki_page = wiki.page(search_term)
    try:
        if wiki_page.exists():
            play_voice_assistant_speech(translator.get("Here is what I found for {} on Wikipedia").format(search_term))
            webbrowser.get().open(wiki_page.fullurl)
            play_voice_assistant_speech(wiki_page.summary.split(".")[:2])
        else:
            play_voice_assistant_speech(translator.get(
                "Can't find {} on Wikipedia. But here is what I found on google").format(search_term))
            url = "https://google.com/search?q=" + search_term
            webbrowser.get().open(url)

    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return


def get_translation(*args: tuple):

    if not args[0]: return

    search_term = " ".join(args[0])
    google_translator = googletrans.Translator()
    translation_result = ""

    old_assistant_language = assistant.speech_language
    try:
        if assistant.speech_language != person.native_language:
            translation_result = google_translator.translate(search_term,
                                                             src=person.target_language,
                                                             dest=person.native_language)

            play_voice_assistant_speech("The translation for {} in Russian is".format(search_term))

            assistant.speech_language = person.native_language
            setup_assistant_voice()

        else:
            translation_result = google_translator.translate(search_term,
                                                             src=person.native_language,
                                                             dest=person.target_language)
            play_voice_assistant_speech("По-английски {} будет как".format(search_term))

            assistant.speech_language = person.target_language
            setup_assistant_voice()

        play_voice_assistant_speech(translation_result.text)

    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()

    finally:
        assistant.speech_language = old_assistant_language
        setup_assistant_voice()


def get_weather_forecast(*args: tuple):

    city_name = person.home_city

    if args:
        if args[0]:
            city_name = args[0][0]

    try:
        weather_api_key = os.getenv("c05560518a428e20ba0329689479ccf32")
        open_weather_map = OWM("342023654c67a2aad521145561ee7ebd")

        weather_manager = open_weather_map.weather_manager()
        observation = weather_manager.weather_at_place(city_name)
        weather = observation.weather

    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    status = weather.detailed_status
    temperature = weather.temperature('celsius')["temp"]
    wind_speed = weather.wind()["speed"]
    pressure = int(weather.pressure["press"] / 1.333)  # переведено из гПА в мм рт.ст.

    print(colored("Weather in " + city_name +
                  ":\n * Status: " + status +
                  "\n * Wind speed (m/sec): " + str(wind_speed) +
                  "\n * Temperature (Celsius): " + str(temperature) +
                  "\n * Pressure (mm Hg): " + str(pressure), "yellow"))

    play_voice_assistant_speech(translator.get("It is {0} in {1}").format(status, city_name))
    play_voice_assistant_speech(translator.get("The temperature is {} degrees Celsius").format(str(temperature)))
    play_voice_assistant_speech(translator.get("The wind speed is {} meters per second").format(str(wind_speed)))
    play_voice_assistant_speech(translator.get("The pressure is {} mm Hg").format(str(pressure)))


def change_language(*args: tuple):

    assistant.speech_language = "ru" if assistant.speech_language == "en" else "en"
    setup_assistant_voice()
    print(colored("Language switched to " + assistant.speech_language, "cyan"))


def run_person_through_social_nets_databases(*args: tuple):
    if not args[0]: return

    google_search_term = " ".join(args[0])
    vk_search_term = "_".join(args[0])
    fb_search_term = "-".join(args[0])

    url = "https://google.com/search?q=" + google_search_term + " site: vk.com"
    webbrowser.get().open(url)

    url = "https://google.com/search?q=" + google_search_term + " site: facebook.com"
    webbrowser.get().open(url)

    vk_url = "https://vk.com/people/" + vk_search_term
    webbrowser.get().open(vk_url)

    fb_url = "https://www.facebook.com/public/" + fb_search_term
    webbrowser.get().open(fb_url)

    play_voice_assistant_speech(translator.get("Here is what I found for {} on social nets").format(google_search_term))


def toss_coin(*args: tuple):

    flips_count, heads, tails = 3, 0, 0

    for flip in range(flips_count):
        if random.randint(0, 1) == 0:
            heads += 1

    tails = flips_count - heads
    winner = "Tails" if tails > heads else "Heads"
    play_voice_assistant_speech(translator.get(winner) + " " + translator.get("won"))


def how_are(*args: tuple):
    l = "Good how are you doing"
    play_voice_assistant_speech(translator.get(l))


def name(*args: tuple):
    l = "My name is djon"
    play_voice_assistant_speech(translator.get(l))

def goodd(*args: tuple):
    l = "I'm glad, how can I help?"
    play_voice_assistant_speech(translator.get(l))


def Eva(*args: tuple):
    l = "recognize and synthesize speech without Internet access report weather forecasts anywhere in the world make a search query in a search engine and also open a list of results and the results of this query make a search query for a video in the YouTube system and open a list of results of this query; search for a definition in Wikipedia with further reading of the first two sentences; search for a person by first and last name in the social networks VKontakte and Facebook; flip a coin translate from the language being studied to the user's native language (taking into account the peculiarities of speech reproduction) play a random greeting; play a random farewell with subsequent termination of the program change the settings of the speech recognition and synthesis language"
    play_voice_assistant_speech(translator.get(l))




config = {
    "intents": {
        "greeting": {
            "examples": ["привет", "здравствуй", "добрый день",
                         "hello", "good morning"],
            "responses": play_greetings
        },
        "farewell": {
            "examples": ["пока", "до свидания", "увидимся", "до встречи",
                         "goodbye", "bye", "see you soon"],
            "responses": play_farewell_and_quit
        },
        "google_search": {
            "examples": ["найди в гугл",
                         "search on google", "google", "find on google"],
            "responses": search_for_term_on_google
        },
        "youtube_search": {
            "examples": ["найди видео", "покажи видео",
                         "find video", "find on youtube", "search on youtube"],
            "responses": search_for_video_on_youtube
        },
        "wikipedia_search": {
            "examples": ["найди определение", "найди на википедии",
                         "find on wikipedia", "find definition", "tell about"],
            "responses": search_for_definition_on_wikipedia
        },
        "person_search": {
            "examples": ["пробей", "найди человека",
                         "find on facebook", " find person", "run person", "search for person"],
            "responses": run_person_through_social_nets_databases
        },
        "weather_forecast": {
            "examples": ["прогноз погоды", "какая погода","погода"
                         "weather forecast", "report weather"],
            "responses": get_weather_forecast
        },
        "translation": {
            "examples": ["выполни перевод", "переведи",
                         "translate", "find translation"],
            "responses": get_translation
        },
        "language": {
            "examples": ["смени язык", "поменяй язык",
                         "change speech language", "language"],
            "responses": change_language
        },
        "toss_coin": {
            "examples": ["подбрось монетку", "подкинь монетку",
                         "toss coin", "coin", "flip a coin"],
            "responses": toss_coin
        },
        "How_are_you": {
            "examples": ["как у тебя дела", "как настроение",
                         "how are you", "are you ok" ,"farewell"],
            "responses": how_are
        },
        "Name": {
            "examples": ["как тебя зовут", "как твоё имя",
                         "what you name", "you name is"],
            "responses": name
        },
        "Good": {
            "examples": ["отлично" , "плохо",
                         "good", "nice" , "so so"],
            "responses": goodd
        },
        "what can you do?": {
            "examples": ["функции" ,"знания",
                         "what can you do"],
            "responses": Eva
        }

    },

    "failure_phrases": play_failure_phrase
}


def prepare_corpus():

    corpus = []
    target_vector = []
    for intent_name, intent_data in config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            target_vector.append(intent_name)

    training_vector = vectorizer.fit_transform(corpus)
    classifier_probability.fit(training_vector, target_vector)
    classifier.fit(training_vector, target_vector)


def get_intent(request):

    best_intent = classifier.predict(vectorizer.transform([request]))[0]

    index_of_best_intent = list(classifier_probability.classes_).index(best_intent)
    probabilities = classifier_probability.predict_proba(vectorizer.transform([request]))[0]

    best_intent_probability = probabilities[index_of_best_intent]

    print(best_intent_probability)
    if best_intent_probability > 0.110:
        return best_intent


def make_preparations():

    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, classifier

    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    ttsEngine = pyttsx3.init()

    person = OwnerPerson()
    person.name = "Maria"
    person.home_city = "Tyumen"
    person.native_language = "ru"
    person.target_language = "en"

    assistant = VoiceAssistant()
    assistant.name = "kaysi"
    assistant.sex = "female"
    assistant.speech_language = "ru"

    setup_assistant_voice()

    translator = Translation()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()





if __name__ == "__main__":
    make_preparations()

    while True:
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        if voice_input:
            voice_input_parts = voice_input.split(" ")

            if len(voice_input_parts) == 1:
                intent = get_intent(voice_input)
                if intent:
                    config["intents"][intent]["responses"]()
                else:
                    config["failure_phrases"]()


            if len(voice_input_parts) > 1:
                for guess in range(len(voice_input_parts)):
                    intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    print(intent)
                    if intent:
                        command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        print(command_options)
                        config["intents"][intent]["responses"](*command_options)
                        break
                    if not intent and guess == len(voice_input_parts) - 1:
                        config["failure_phrases"]()

