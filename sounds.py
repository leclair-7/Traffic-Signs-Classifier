#beep1.wav

def bing():
	import pygame
	import time

	pygame.mixer.init() #frequency, size, channels, buffersize | 44100, 16, 2, 4096
	soundObj = pygame.mixer.Sound('beep1.ogg')
	soundObj.play()

	time.sleep(2)
	soundObj.stop()

bing()

look into cross validation, sklearn
