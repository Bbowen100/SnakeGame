import pygame, random, sys
from pygame.locals import *
from nn import neural_net
import numpy as np
import math as m

class SnakeGame():

	def __init__(self, epoch=10, batch_size=10, epsilon=0.3, gamma=.8):
		self.epoch = epoch
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.gamma = gamma
		self.model = neural_net([300, 300])
		self.experience = []

	# Check to see if there is a collision with a wall/apple/neither between two objects
	def collide(self, Object1x, Object2x, Object1y, Object2y, Object1Width, Object2Width, Object1Height, Object2Height):
		if Object1x+Object1Width>Object2x and Object1x<Object2x+Object2Width and Object1y+Object1Height>Object2y and Object1y<Object2y+Object2Height:return True
		else:return False

	def die(self, screen, score):
		f=pygame.font.SysFont('Arial', 30);
		t=f.render('Your score was: '+str(score), True, (0, 0, 0));
		screen.blit(t, (10, 270));
		pygame.display.update();
		pygame.time.wait(500);
		sys.exit(0)

	def getNewState(self, oldState, action, dirs=0):
		newState = oldState[:]
		if(action == 2 and dirs != 0):
			dirs = 2
		elif(action == 0 and dirs !=2):
			dirs = 0
		elif(action == 3 and dirs !=1):
			dirs = 3
		elif(action == 1 and dirs !=3):
			dirs = 1

		if(dirs == 0):
			newState[1] += 20
		elif(dirs == 1):
			newState[0] += 20
		elif(dirs == 2):
			newState[1] -= 20
		elif(dirs == 3):
			newState[0] -= 20
		return newState

	def getState(self):
		# print(self.xs[0], self.ys[0], self.applepos[0], self.applepos[1])
		return([self.xs[0], self.ys[0], self.applepos[0], self.applepos[1]])

	def distance(self, state):
		SnakeX = state[0]; SnakeY = state[1];
		AppleX = state[2]; AppleY = state[3];
		d = m.sqrt(m.pow((AppleX - SnakeX),2) + m.pow((AppleY - SnakeY),2))
		return d

	def reward(self, oldState, action, dirs):
		newState = self.getNewState(oldState, action, dirs)
		# -500 for restarting the game
		if(self.collide_self_wall(newState)):
			return -500
		# reward +10 if snake is closer to apple, -10 if snake is farther
		# and +100 if the snake gets the apple
		oldDistance = self.distance(oldState)
		newDistance = self.distance(newState)
		if(oldDistance > newDistance):
			if(newDistance == 0):
				return 100
			else:
				return 10
		elif(oldDistance < newDistance):
			return -10
		else:
			return 0 # same spot: Unlikely but for debugging purposes

	def collide_self_wall(self, state):
		SnakeX = state[0]
		SnakeY = state[1]
		#collided with itself
		i = len(self.xs)-1
		collided_w_itself = False
		while i >= 2:
			if self.collide(SnakeX, self.xs[i], SnakeY, self.ys[i], 20, 20, 20, 20):
				return(True)
			i-= 1
		# collide with wall
		if (SnakeX < 0 or SnakeX > 290 or SnakeY < 0 or SnakeY > 290):
			return(True)
		return False

	def collectExperience(self, experience):
		if(experience not in self.experience):
			self.experience.append(experience)

	def playGame(self):
		model = self.model
		self.xs = [150, 150];
		self.ys = [150, 150];
		dirs = 0;
		score = 0;
		self.applepos = (random.randint(0, 290), random.randint(0, 290));
		pygame.init();
		screen=pygame.display.set_mode((300, 300));
		pygame.display.set_caption('SnakeGame');
		Snake = pygame.Surface((20, 20));
		Snake.fill((0, 0, 0));
		appleimage = pygame.Surface((10, 10));
		appleimage.fill((255, 0, 0));
		f = pygame.font.SysFont('Arial', 20);
		clock = pygame.time.Clock()

		frame = 0
		frameRate = 20
		action = 0
		while (True):
			clock.tick(frameRate)

			if(action == 2 and dirs != 0):
				dirs = 2
			elif(action == 0 and dirs !=2):
				dirs = 0
			elif(action == 3 and dirs !=1):
				dirs = 3
			elif(action == 1 and dirs !=3):
				dirs = 1

			for e in pygame.event.get():
			    if e.type == QUIT:
			        sys.exit(0)

			# Decrease epsilon over the first half of training
			# if (self.epsilon > 0.1):
			# 	self.epsilon -= (0.9 / self.epoch)

			# decide which direction the snake will go
			if ((random.random() < self.epsilon) and (frame < self.batch_size)):
				action = random.choice([0,1,2,3]) #take a random direction
			else:
				# get action prediction from the model
				state = np.array(self.getState())
				prediction = model.predict(np.array([state])).flatten().tolist()
				action = prediction.index(max(prediction))

			# get the reward for the action taken with the state
			state = self.getState()
			reward = self.reward(state, action, dirs)

			# get data to record as experience
			predOutput = model.predict(np.array([state])).flatten().tolist()
			newState = self.getNewState(state, action, dirs)
			newStatePrediction = model.predict(np.array([newState])).flatten().tolist()
			predOutput[action] = reward
			experience = [state, predOutput]

			self.collectExperience(experience) # record experience

			# train nueral net on the experience collected
			if(frame == self.batch_size):
				# get training set from experience
				Xtrain = [];Ytrain = [];
				loss = 0
				for ele in self.experience:
					Xtrain.append(ele[0])
					Ytrain.append(ele[1])

				loss = model.fit(np.array(Xtrain), np.array(Ytrain),
				batch_size=len(self.experience), nb_epoch=self.epoch)
				# reset frames and expereince
				frame = 0
				self.experience = []

			# checks if snake collides with itself
			i = len(self.xs)-1
			collided_w_itself = False
			while i >= 2:
				if self.collide(self.xs[0], self.xs[i], self.ys[0], self.ys[i], 20, 20, 20, 20):
					# die(screen, score)
					collided_w_itself = True
				i-= 1
			if collided_w_itself:
				#reset the game
				self.xs = [150, 150];
				self.ys = [150, 150];
				score = 0;

			# checks if snake collides with apple
			if self.collide(self.xs[0], self.applepos[0], self.ys[0], self.applepos[1], 20, 10, 20, 10):score+=1;self.xs.append(700);self.ys.append(700);self.applepos=(random.randint(0,290),random.randint(0,290));
			# check if snake collides with wall
			if (self.xs[0] < 0 or self.xs[0] > 290 or self.ys[0] < 0 or self.ys[0] > 290):
				# die(screen, score)
				# reset the game to beginning
				self.xs = [150, 150];
				self.ys = [150, 150];
				score = 0;

			i = len(self.xs)-1
			# propogates x and y cordinates backwards
			while i >= 1:
				self.xs[i] = self.xs[i-1];
				self.ys[i] = self.ys[i-1];
				i -= 1
			# updates the co-ordinates of the head which will be propogated backwards
			if dirs==0:self.ys[0] += 20
			elif dirs==1:self.xs[0] += 20
			elif dirs==2:self.ys[0] -= 20
			elif dirs==3:self.xs[0] -= 20
			screen.fill((255, 255, 255))
			# print the snake onto the screen
			for i in range(0, len(self.xs)):
				screen.blit(Snake, (self.xs[i], self.ys[i]))

			screen.blit(appleimage, self.applepos);
			t=f.render(str(score), True, (0, 0, 0));
			screen.blit(t, (10, 10));
			pygame.display.update()
			frame+=1

if __name__ == '__main__':
	SnakeGame = SnakeGame(epoch=1)
	SnakeGame.playGame()
