import pygame
import random
import numpy

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400


PADDLE_WID = 16
PADDLE_HEI = 50

BALL_WID = 10
BALL_HEI = 10

PADDLE_SPEED = 2
BALL_X_SPE = 3
BALL_Y_SPE = 2

WHITE = (255 , 255 , 255)
WHITE = (255 , 0 , 0)
BLACK = (0 , 0 , 0)

#init screen
screen = pygame.display.set_mode(WINDOW_WIDTH , WINDOW_HEIGHT)

def drawBall(bX , bY):
    b = pygame.rect(bX , bY , BALL_WID , BALL_HEI )

    pygame.draw.rect(screen , WHITE , b)



def drawPad1(pY):
    p = pygame.rect(PADDLE_BUFFER , pY , PADDLE_WID , PADDLE_HEI )
    pygame.draw.rect(screen , WHITE , p)


def drawPad2(pY):
    p = pygame.rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WID , pY , PADDLE_WID , PADDLE_HEI )
    pygame.draw.rect(screen , WHITE , p)


def upadateBall(pY1 , pY2 , bX , bY , bXD , bYD):
    bX = bX + bXD * BALL_X_SPE 
    bY = bY + bYD * BALL_Y_SPE
    score = 0 

    # Collsion
    if(bX <= PADDLE_BUFFER + PADDLE_WID and bY + BALL_HEI >= p1y and 
       bY - BALL_HEI <= p1y + PADDLE_HEI):
        bXD = 1
    elif(bX <= 0):
        bXD = 1
        score = -1
        return (score , bX , bY , bXD , bYD)

    if(bX >= WINDOW_WIDTH - PADDLE_WID - PADDLE_BUFFER  and bY + BALL_HEI >= p2y and
            bY - BALL_HEI <= p2y + PADDLE_HEI ):
        bXD = - 1
    elif( bX >= WINDOW_WIDTH - BALL_WID):
        bXD = -1 
        score = 1
        return (score , bX , bY , bXD , bYD)
    

    if(bY <= 0):
        bY = 0 
        bYD = 1 
    elif(bY >= WINDOW_HEIGHT - BALL_HEI)
        bY = WINDOW_WIDTH - BALL_HEI
        bYD = -1
    
    return (score , bX , bY , bXD , bYD)



def updatePad(act , p1y):
    if(act[1] == 1):
         p1y = p1y - PADDLE_SPEED
    if(act[2] == 1):
        p1y = p1y + PADDLE_SPEED

    
    if(p1y < 0):
        p1y = 0 
    if(p1y > WINDOW_HEIGHT - PADDLE_HEI):
        p1y = WINDOW_WIDTH - PADDLE_HEI

    return p1y




class PongGame:
    def __init__(self):
        num = random.randint(0,9)
        self.score = 0 

        self.p1y = WINDOW_HEIGHT / 2 - PADDLE_HEI / 2
        self.p2y = WINDOW_HEIGHT / 2 - PADDLE_HEI / 2

        self.bXD = 1 ;
        self.bYD = 1

        self.bX = WINDOW_WIDTH / 2- BALL_WID / 2
        self.bY = WINDOW_HEIGHT / 2 - BALL_HEI / 2

    def getPresentFrame(self):
        pygame.event.pump()
        screen.fill(BLACK)
        drawPad1(self.p1y)
        drawPad2(self.p2y)
        drawBall(self.bX , self.bY)


        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return image_data

    def getNextFrame(self , action):
        pygame.event.pump()
        screen.fill(BLACK)
        self.p1y = updatePad(action , self.p1y)
        drawPad1(self.p1y)
        self.p2y = updatePad(action , self.p2y)
        drawPad2(self.p2y) 
        (score , self.bX , self.by , self.bXD , self.bYD) = upadateBall(self.p1y . self.p2y , self.bX , self.bY , self.bXD , self.bYD)

        drawBall(self.bX , self.bY)
         image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return (score , image_data)


        
        
        
