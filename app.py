import sys
import cv2
import pygame
import numpy as np
# import cvzone as cvz
import mediapipe as mp
from csv import reader
from colors import Colors as C
from imutils.video import VideoStream
from math import floor

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



font_family = 'gadugi.ttf'
camera_index = 0
question_file = 'questions.csv'

class CameraViewer:

    current_question = 0
    selected_optopn_index = 0
    correct_option = 'A'
    

    def __init__(self, camera_index=0):

        pygame.init()
        # full screen
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.vs = VideoStream(src=camera_index).start()
        self.clock = pygame.time.Clock()
        self.questions = self.read_question(question_file)
        self.hands = mp_hands.Hands( min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def read_question(self, csvfile='questions.csv'):
        '''read questions from csv file'''
        with open(csvfile, 'r', encoding='UTF-8') as read_obj:
            csv_reader = reader(read_obj)
            return list(csv_reader)[1:]
        
    def get_question(self, index):
        '''returns question and options'''
        return self.questions[index]

    def modify_video(self, frame):
        # double the size of the frame
        frame = cv2.resize(frame, (0, 0), fx=2.2, fy=2.2)
        frame = np.rot90(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reduce the opacity of the frame .2
        frame = cv2.addWeighted(frame, .2, np.zeros(frame.shape, frame.dtype), 0, 0)
        return frame
        
    
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    def put_text(self, text, pos, fontsize=32, text_color=C.WHITE, background=C.BLACK,centered=True):
        font = pygame.font.Font('gadugi.ttf', fontsize)
        text = font.render(text, True, text_color, background)
        if centered:
            text_rect = text.get_rect(center=pos)
        else:
            text_rect = text.get_rect(topleft=pos)
        self.screen.blit(text, text_rect)
        return text_rect

    def put_button(self, option_text, pos,width=400, fontsize=50, color=C.BLUE):
        btn = pygame.Rect(pos, (width-40, 100))
        pygame.draw.rect(self.screen, color , btn, border_radius=10)
        font = pygame.font.Font('gadugi.ttf', fontsize)
        text = font.render(option_text, True, C.YELLOW, color)
        text_rect = text.get_rect(center=btn.center)
        # reduce text size if it is too long
        while text_rect.width > btn.width - 20:
            fontsize -= 1
            font = pygame.font.Font('gadugi.ttf', fontsize)
            text = font.render(option_text, True, C.YELLOW, color)
            text_rect = text.get_rect(center=btn.center)
        self.screen.blit(text, text_rect)

    def ui(self,screen_width, frame):
        '''shows question and options to the user for gesture based control'''
        # draw a rectangle after 10 px below the frame
        question, oA, oB, oC, oD, ans= qobj = self.get_question(self.current_question)
        self.correct_option = ans
        question_box = pygame.Rect((screen_width // 2 - frame.get_width() // 2), frame.get_height() + 30, frame.get_width(), 150)
        pygame.draw.rect(self.screen, C.BLACK, question_box, border_radius=10)
        # put the question in the rectangle
        self.put_text(question, (question_box.centerx, question_box.centery), fontsize=50, centered=True)
        # put options inside the frame area from top 100
        option_box = pygame.Rect((screen_width // 2 - frame.get_width() // 2),200, frame.get_width(), 500)
        # transparent rectangle
        transparent_color = (0, 0, 0, 0)
        shape_surf = pygame.Surface(pygame.Rect(option_box).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, transparent_color, option_box)
        # print(option_box.x, option_box.y)
        gap = 150
        self.put_button(oA, (option_box.x + 20, option_box.y + 20), width=option_box.w)
        self.put_button(oB, (option_box.x + 20, option_box.y + 20 + gap), width=option_box.w)
        self.put_button(oC, (option_box.x + 20, option_box.y + 20 + gap * 2), width=option_box.w)
        self.put_button(oD, (option_box.x + 20, option_box.y + 20 + gap * 3), width=option_box.w)

    def check_answer(self):
        '''check if the selected option is correct'''
        if self.selected_optopn_index == ord(self.correct_option) - ord('A'):
            return True
        else:
            return False
        
    def next_question(self):
        '''go to next question'''
        self.current_question += 1
        self.selected_optopn_index = None
        self.correct_option = None

    def detect_hand(self, image):
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        data = {'image':image, 'index_finger':None, 'middle_finger':None}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # get index finger tip
                index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                middle_x, middle_y = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
                # normalize the coordinates
                index_x = min(floor(index_x * image.shape[1]), image.shape[1] - 1)
                index_y = min(floor(index_y * image.shape[0]), image.shape[0] - 1)
                middle_x = min(floor(middle_x * image.shape[1]), image.shape[1] - 1)
                middle_y = min(floor(middle_y * image.shape[0]), image.shape[0] - 1)
                # put in dictionary
                data['index_finger'] = (index_x, index_y)
                data['middle_finger'] = (middle_x, middle_y)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
        return data

    def draw_circle(self, coord):
        if coord:
            # draw a traslucent circle
            width, height = 100,100
            surface = pygame.Surface((width,height), pygame.SRCALPHA)
            pygame.draw.circle(surface, (255, 0, 0, 100), coord, 50)


    
        
    def run(self):
        screen_width, screen_height = pygame.display.get_surface().get_size()
        pygame.display.set_caption('Augmented Reality Quiz')

        while True:
            image = self.vs.read()
            image = self.modify_video(image)
            
            data = self.detect_hand(image)
            image = data['image']
            index_finger = data['index_finger']
            middle_finger = data['middle_finger']
            frame = pygame.surfarray.make_surface(image)
           
            self.draw_circle(index_finger)
            self.draw_circle(middle_finger)

            self.screen.fill(C.WHITE)
            # put video in the center of screen
            self.screen.blit(frame, ((screen_width // 2 - frame.get_width() // 2), 20))
            # display UI
            self.ui(screen_width,frame)
            
            

            # hand detection

            # last lines of code
            self.handle_events()
            self.clock.tick(60)
            pygame.display.update()




        
           

        

if __name__ == '__main__':
    viewer = CameraViewer()
    viewer.run()
