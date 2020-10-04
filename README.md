# Snake-AI
Very simple Snake AI using Neural Networks and Genetic Algorithms, in the creation of this no Snakes were hurt.
i created this using tensorflows' keras and very basic genetic Algorithms, pygame is used for the graphical output of the snake, the snake game is the one i made earlier: https://github.com/andwehrm/python-snake only now the AI is playing and not me.

![Alt text](https://raw.githubusercontent.com/andwehrm/Snake-AI/master/demo.gif "Demonstration of Gen 10")


After 10 Generations (10000 Games Played) this is what one of the Snakes plays like, as you can see it has developed its own simple strategy which short term works out pretty well but as soon as the Snake gets too long it has no chance to get any further with this strategy.
The training process with this AI takes very long since there is only one Snake playing at all times, to counter this you could add some Multi-Threading to have multiple Snakes training at once. Also the Snake abuses a small bug in the collision detection, it goes off the screen because i accidentally have the collision not quite perfect.


![Alt text](https://raw.githubusercontent.com/andwehrm/Snake-AI/master/Neural%20Network.png "Neural Network")

This is the Model i used, as you can see it is a simple Sequential Model will just Dense layers, for activation function on the Hidden Layer i used a Linear function and the Activation function for the Output Layer is using Sigmoid, playing arround with this might yield better results but using this i could create a quite fitting model
Considering the limited view the Snake has i think it has created a quite fitting method as it just slithers arround the whole playing field thus always finding the food no matter where it is, the only problem with this technique is that as the Snake grows longer it will block of its way back with its own Tail. To Counter this one would probably need to give the Snake more complex information about its surroundings.
I am happy with the result though since this is my first time creating an AI using Neural Networks and Machine Learning, this project was meant for me to learn something and i definetly did.



