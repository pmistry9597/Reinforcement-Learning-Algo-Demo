import gymnasium as gym
import random

import tkinter as tk
from PIL import ImageTk, Image

############# This file is meant to be a basic, interactive session with the environment and an agent
# user input can be selected, or a different agent can be chosen. The latter case this is just a real-time(ish) session of watching an agent.
# Also, I said basic, coding conventions here might suck :P

env = gym.make('LunarLander-v2', render_mode="rgb_array")

obs0, info0 = env.reset()
h, w, chans = env.render().shape

# windowing system setup
mwin = tk.Tk()
mwin.config(width=800, height=600)
mwin.title("yourmom")
c = tk.Canvas(mwin, height = h, width = w)
c.place(x=(800 - w) / 2, y=(600 - h) / 2)

print("observation set:", env.observation_space)
print("action set:", env.action_space)

# key press system, which is active but does not have to interfere with the env unless chosen by changing the agent type

action = 0
pressn = 0
def press(e):
    global action, pressn
    if e.char == 'w':
        action = 2
    if e.char == 'd':
        action = 3
    if e.char == "a":
        action = 1
    pressn += 1
def release(e):
    global action, pressn
    pressn += -1
    if pressn == 0:
        action = 0

mwin.bind("<KeyPress>", press)
mwin.bind("<KeyRelease>", release)

def user_action(_obs):
    return action

# agent selection/setup
from rl import actor

# agent = actor.RandomActor(env.action_space)
agent = actor.ExternalActor(user_action) # user agent case

obs = obs0 #obs will be observation variable used to store previously observed state to choose next action

for i in range(400):
    act = agent(obs)
    obs, reward, termin, trunc, info  = env.step(act) #take action, advance environment
    # print(reward)
    print(obs)

    im = ImageTk.PhotoImage(Image.fromarray(env.render()))
    imtk = c.create_image(0,0,anchor="nw", image=im)

    mwin.update_idletasks()
    mwin.update()