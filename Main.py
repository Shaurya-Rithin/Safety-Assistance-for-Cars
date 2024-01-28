from keras.models import load_model 
import cv2  
import numpy as np
import pyttsx3
import pygame

pygame.init()

surface = pygame.display.set_mode((700, 600))

color = (255, 255, 255)

surface.fill(color)
pygame.display.flip()

imp = pygame.image.load("road.png").convert()
car = pygame.image.load("car.png").convert()
car = pygame.transform.scale(car, (200, 100))
person = pygame.image.load("person.png").convert()
person = pygame.transform.scale(person, (50, 100))
br = pygame.image.load("br.png").convert()
br = pygame.transform.scale(br, (50, 100))
sb = pygame.image.load("spedbumb.png").convert()
sb = pygame.transform.scale(sb, (200, 200))

engine = pyttsx3.init()

def talk(text):
    engine.say(text)
    print(text)
    engine.runAndWait()

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    surface.blit(imp, (0,100))
    surface.blit(car, (0,150))
    extval, image = camera.read()

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    #cv2.imshow("Webcam Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    class_name = class_names[index].strip() 
   
    if class_name == "3 walls":
        surface.blit(br, (500,150))
        talk("wall")
    #if class_name == "0 Empty Road you can go":
        #r
    if class_name == "1 Speed Bump slow down":
        talk("Speed Bump")  
        surface.blit(sb, (500,150))
    if class_name == "2 Pedestrian STOP":
        surface.blit(person, (500,150))
        talk("person")
        

    pygame.display.update()