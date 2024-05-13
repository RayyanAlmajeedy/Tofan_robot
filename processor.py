import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('job_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

import openai

api_key = ""
openai.api_key = api_key
#client = OpenAI(api_key=OPENAI_API_KEY)

def chat_bot(prompt):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {"role": "system", "content": "You are a tofan robot."},
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    print("المطلوب",tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            #result = "You must ask the right questions"
            #result = chat_bot(ints)
            result = 0
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    tag = float(ints[0]['probability'])
    #i = ints
    print(tag)
    if tag <=0.6:
        res = chat_bot(msg)
    else:
        res = getResponse(ints, intents)
    return res




#######################
################
########

from gtts import gTTS
from playsound import playsound
import  speech_recognition as sr
import os
import pygame
"""""
def play_sound(sound_file):
    try:
        # تهيئة Pygame
        pygame.init()
        # تحميل الملف الصوتي
        pygame.mixer.music.load(sound_file)
        # تشغيل الملف الصوتي
        pygame.mixer.music.play()
        # انتظار حتى انتهاء التشغيل
        while pygame.mixer.music.get_busy():
            continue
    except pygame.error as e:
        print(f"حدث خطأ: {e}")
    finally:
        pygame.quit()

def convert_text_to_speech(text, lang='ar'):
    tts = gTTS(text=text, lang=lang)
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")
    tts.save("output.mp3")
    # استدعاء الدالة وتمرير اسم الملف الصوتي كمعامل
    play_sound("output.mp3")
   """

def play_sound(sound_file):
    try:
        pygame.init()
        pygame.mixer.music.load(sound_file)
        print("جاري تشغيل الصوت...")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except pygame.error as e:
        print(f"حدث خطأ: {e}")
    finally:
        pygame.quit()

def convert_text_to_speech(text, lang='ar'):
    tts = gTTS(text=text, lang=lang)
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")
    tts.save("output.mp3")
    play_sound("output.mp3")

   ####################
def convert():
    """ دالة التحقق اذا تم التعرف على الصوت"""
    r = sr.Recognizer()
    with sr.Microphone() as src:
        try:
            audio = r.listen(src)
            t = r.recognize_google(audio, language='ar-AR')
            return t
        except sr.UnknownValueError as U:
            print("خطأ: لم يتم استقبال صوت من الميكروفون")
            return None
        except sr.RequestError as R:
            print("خطأ في الطلب:", R)
            return None
              
def convert_speech_to_text():
    result = convert()
    if result is not None:
       return result
    else:
       r=1  
       return r

#################
"""for i in range(10):
    print("تكلم")
    question =convert_speech_to_text()
    if question == "خروج":
        break
    if question==1:  # إذا كان النص فارغاً
        # تخطي هذه الدورة من الحلقة والانتقال إلى التكرار التالي
        continue
    response = chatbot_response(question)
    if response == 0 :
        res = chat_bot(question)
        print(res)
        convert_text_to_speech(res)
    else :
        print(response) 
        convert_text_to_speech(response)

"""


#####################3

###############
while True:
    keyword=["طوفان","مرتضى","يوسف","ريان","بسام","عمر"]
    input_key=convert_speech_to_text()
    print(input_key)
    #input_key=input("ادخل الكلمة المفتاحية : ")
    if input_key in keyword:
        while True:
            question = convert_speech_to_text()
            if question == "خروج":
                x="وداعا"
                convert_text_to_speech(x)
                break
            if question==1:  # إذا كان النص فارغاً
            # تخطي هذه الدورة من الحلقة والانتقال إلى التكرار التالي
                continue
            response = chatbot_response(question)
            print(response)
            convert_text_to_speech(response)
            """
            if response == 0:
                 res = chat_bot(question)
                 print(res)
            else :
                 print(response)"""
    else:
        h="أدخل الكلمة المفتاحية"
        convert_text_to_speech(h)

    
    
    
    
                                

