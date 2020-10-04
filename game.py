import itertools
import random
import time
from math import sqrt
from random import randint, uniform
import threading
import matplotlib.pyplot as plt
import numpy as np
import pygame
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

def model_mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if( uniform(0,1) > .80):
                change = uniform(-1,1)
                weights[i][j] = change
    return weights

def model_create():
    model = Sequential([
    Dense(units=18, input_shape=(8,), activation="linear"),
    Dense(units=4, activation='sigmoid',input_shape=(8,))
    ])

    model.compile(optimizer='adam',loss='mse')
    
    return model


last_dist = 1000000000
# Game Settings
max_moves = 200
current_moves = 0
dis_width = 150
dis_height = 150
block_size = 10
snake_size = 1
snake_speed = 60
food_strength = 1
snake_color = [220,61,75]
food_color = [76,217,100]
background_color = [28,37,46]
start_point = [0,0]
text_color = [220,61,75]
model = model_create()
models = []
fits = []
num = 0
generation = 0

for i in range(1000):
    models.append(model_create())
    fits.append(0)

# Internal Values
x_food = 0
y_food = 0
x_food2 = 0
y_food2 = 0
x = start_point[0]
y = start_point[1]
y_change = 0
x_change = 0
snake_list = []
snake_block = []
moved = False
best_fit = 0
best_weights = model.get_weights()

def draw_text(msg,font,color,coords, center):
    mesg = font.render(msg, True, color)
    if center == True:
        text_rec = mesg.get_rect(center=(coords[0]/2,coords[1]/2))
        dis.blit(mesg,text_rec)
    elif center == False:
        dis.blit(mesg,[coords[0],coords[1]])


def spawn_food():
    global x_food
    global y_food
    global x_food2
    global y_food2
    x_food = random.randrange(0,(dis_width-block_size)/block_size)*block_size
    y_food = random.randrange(0,(dis_height-block_size)/block_size)*block_size
    if(x_food == x and y_food == y):
        spawn_food()
    x_food2 = random.randrange(0,(dis_width-block_size)/block_size)*block_size
    y_food2 = random.randrange(0,(dis_height-block_size)/block_size)*block_size
    if(x_food2 == x and y_food2== y):
        spawn_food()    


def Distance(x1,y1,x2,y2):
    return sqrt( (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def model_crossover(parent1, parent2):
    global models

    weight1 = models[parent1].get_weights()
    weight2 = models[parent2].get_weights()

    new_weight1 = weight1
    new_weight2 = weight2

    gene = randint(0,len(new_weight1)-1)

    new_weight1[gene] = weight2[gene]
    new_weight2[gene] = weight1[gene]

    return np.asarray([new_weight1,new_weight2])

def new_generation():
    global num
    global best_weights
    global best_fit
    global generation
    spawn_food()
    num = 0
    new_weights = []
    print('New Generation')
    generation += 1
    parent1 = randint(0,999)
    parent2 = randint(0,999)

    for fit in fits:
        print(fit)

    for i in range(1000):
        if fits[i] > best_fit:
           best_fit = fits[i]
           best_weights = models[i].get_weights()

    for i in range(1000):
        if fits[i] > fits[parent1]:
            parent1 = i

    for i in range(1000):
        if i != parent1:
            if fits[i] > fits[parent2]:
                parent2 = i

    print("Parent1:", parent1)
    print("Fitness:", fits[parent1])

    print("Parent2:", parent2)
    print("Fitness:", fits[parent2])

    for select in range(1000//2):
        cross_over_weights = model_crossover(parent1,parent2)

        mutated1 = model_mutate(models[parent1].get_weights())
        mutated2 = model_mutate(models[parent2].get_weights())   

        new_weights.append(mutated1)
        new_weights.append(mutated2)

    for select in range(len(new_weights)):
        if fits[select] < best_fit//5:
            models[select].set_weights(new_weights[select])
        fits[select] = 0
    
    models[0].set_weights(best_weights)

    


 

def game_end():
    global current_moves
    global x
    global y
    global snake_size
    global start_point
    global snake_list
    global last_food_x
    global last_food_y
    global x_food
    global y_food
    global model
    global num
    global last_dist

    current_moves = 0

    last_dist = 100000000
    head = snake_list[0]
    snake_list = []
    snake_list.append(head)
    
    x = start_point[0]
    y = start_point[1]
    snake_size = 1

    print(fits[num])

    num += 1
    if num >= 1000:
        new_generation()



    



    




pygame.init()
font_style = pygame.font.SysFont("ubuntu",25)
font_score = pygame.font.SysFont("ubuntu",18)

dis=pygame.display.set_mode((dis_width,dis_height))
pygame.display.update()
pygame.display.set_caption('Snake')


clock = pygame.time.Clock()
spawn_food()
spawn_food()


model = model_create()

last_move = -1
def get_fit_for_pos(new_x,new_y):
    new_fit = 0
    global last_dist
    global x_food
    global y_food
    global x_food2
    global y_food2
    global dis_width
    global dis_height

#    if(last_dist > Distance(new_x,new_y,x_food,y_food)):
#        if(Distance(new_x,new_y,x_food,y_food) > 500):
#            new_fit += 1
#        elif(Distance(new_x,new_y,x_food,y_food) < 500):
#            new_fit += 2
#        elif(Distance(new_x,new_y,x_food,y_food) < 400):
#            new_fit += 4
#       elif(Distance(new_x,new_y,x_food,y_food) < 200):
#            new_fit += 32
#        elif(Distance(new_x,new_y,x_food,y_food) < 100):
#            new_fit += 64
#        elif(Distance(new_x,new_y,x_food,y_food) < 50):
#            new_fit += 128
#    else:
#        new_fit -= 1''
#
#    new_fit += 1

    # Out of Bounds detection
    if new_x > dis_width or new_x < 0 or new_y > dis_height or new_y < 0:
        new_fit -= 100000

    # Collision with own Body detection
    for i in snake_list[1:]:
        if new_x == i[0] and new_y == i[1]:
            new_fit -= 100000

    if(x_food == new_x and y_food == new_y):
        new_fit += 1000
    if(x_food2 == new_x and y_food2 == new_y):
        new_fit += 1000    

    return new_fit


def main_game():
    global model
    global last_dist
    global x
    global y
    global x_food
    global y_food
    global fits
    global last_move
    global moved
    global x_change
    global y_change
    global current_moves
    global max_moves
    global dis_width
    global dis_height
    global snake_list
    global snake_size
    global snake_color
    global block_size
    global dis
    global food_strength
    global x_food2
    global y_food2
    global food_color
    global font_score
    global text_color
    global generation
    global snake_speed
    while True:
        model = models[num]
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                game_end()


        if(last_dist > Distance(x,y,x_food,y_food)):
            last_dist = Distance(x,y,x_food,y_food)
            fits[num] += 3
            #if(Distance(x,y,x_food,y_food) > 500):
            #  fits[num] += 1
            #elif(Distance(x,y,x_food,y_food) < 500):
            #  fits[num] += 2
            #elif(Distance(x,y,x_food,y_food) < 400):
            #  fits[num] += 4
        #elif(Distance(x,y,x_food,y_food) < 200):
            #   fits[num] += 32
            #elif(Distance(x,y,x_food,y_food) < 100):
            #   fits[num] += 64
            #elif(Distance(x,y,x_food,y_food) < 50):
        #     fits[num] += 128
        else:
            last_dist = Distance(x,y,x_food,y_food)
            fits[num] -= 1.5
        

        #fits[num] += 1
    
        distance_to_food = Distance(x,y,x_food,y_food)
        distance_left = x
        distance_right = dis_width - x
        distance_top = y
        distance_bot = dis_height - y

        right_fit = get_fit_for_pos(x+10,y)
        left_fit = get_fit_for_pos(x-10,y)
        top_fit = get_fit_for_pos(x,y-10)
        bot_fit = get_fit_for_pos(x,y+10)

        right_free = True
        left_free = True
        top_free = True
        bot_free = True

        right_fruit = False
        left_fruit = False
        top_fruit = False
        bot_fruit = False

        if right_fit < 0 or last_move == 2:
            right_free = False
        if left_fit < 0 or last_move == 3:
            left_free = False
        if top_fit < 0 or last_move == 0:
            top_free = False
        if bot_fit < 0 or last_move == 1:
            bot_free = False
        
        if right_fit > 0:
            right_fruit = True
        if left_fit > 0:
            left_fruit = True
        if top_fit > 0:
            top_fruit = True
        if bot_fit > 0:
            bot_fruit = True
        

        prediction = model.predict(x=[(bot_free,top_free,left_free,right_free,bot_fruit,top_fruit,left_fruit,right_fruit)])

        move = -1 
        val = -1

        for i in range(4):
            if prediction[0][i] > val:

                val = prediction[0][i]
                move = i

        last_move = move

        if move == 0 and moved == True: # Down
                y_change = block_size
                x_change = 0
                moved = False
        elif move == 1 and moved == True: # Up
                y_change = -block_size
                x_change = 0
                moved = False
        elif move == 2 and moved == True: # Left
                x_change = -block_size
                y_change = 0
                moved = False
        elif move == 3 and moved == True: # Right
                x_change = block_size
                y_change = 0
                moved = False




        dis.fill(background_color)


    # Snake Movement
        x += x_change
        y += y_change
        moved = True
        current_moves += 1

        if current_moves > max_moves:
            game_end()

    # Out of Bounds detection
        if x > dis_width or x < 0 or y > dis_height or y < 0:
            game_end()
            continue

    # Collision with own Body detection
        for i in snake_list[1:]:
            if x == i[0] and y == i[1]:
                game_end()
                continue

    # Snake Calculation
        snake_list.append([x,y])

        if len(snake_list) > snake_size:
            del snake_list[0]

    # Drawing the Snake
        for i in snake_list:
            pygame.draw.rect(dis,snake_color,[i[0],i[1],block_size,block_size])


    # Collision with food detection
        for i in snake_list:
            if(x_food == i[0] and y_food == i[1]):
                snake_size+=food_strength
                fits[num] += 150
                current_moves -= 250
                spawn_food()

        for i in snake_list:
            if(x_food2 == i[0] and y_food2 == i[1]):
                snake_size+=food_strength
                fits[num] += 150
                current_moves -= 250
                spawn_food()


    # Drawing the Food
        pygame.draw.rect(dis,food_color,[x_food,y_food,block_size,block_size]) 
        pygame.draw.rect(dis,food_color,[x_food2,y_food2,block_size,block_size]) 
    # Score Display
        draw_text("Score: " + str(snake_size),font_score,text_color,[5,0],False)
        draw_text("Generation: " + str(generation),font_score,text_color,[5,20],False)

        pygame.display.update()
        clock.tick(snake_speed)
threading.Thread(target=main_game).start()



 