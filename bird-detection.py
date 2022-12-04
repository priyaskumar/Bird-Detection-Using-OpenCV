# Importing necessary libraries
import cv2
import time


# Class Definition
class DetectBirds(object):

    # Constructor
    def __init__(self, video, mx_num_birds = 3):
        
        # Getting a video
        self.cap = cv2.VideoCapture(f"Videos/{video}")
        
        # Uncomment the following line for automatic USBCam (0-web cam default)
        # self.cap = cv2.VideoCapture(0) 

        # Definition of haar cascades for bird recognition
        self.birdsCascade = cv2.CascadeClassifier("bird-cascade.xml")
        
        # Validate if ESC key is pressed to close the window
        self.running = True

        # Previous Time Elapsed
        self.pTime = 0

        # Number of birds detected in a frame
        self.num_birds = 0

        # Set minimum number of birds to be detected per frame as 3
        self.MIN_NUM_BIRDS = mx_num_birds

    # Function to detect birds during runtime
    def detect(self):
        while self.running:
            # Capture frame-by-frame from a video
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame into gray scale for better analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect birds in the gray scale image
                birds = self.birdsCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.4,
                    minNeighbors=5,
                    maxSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
                
                # Draw a rectangle around the detected birds in the frame
                for (x, y, w, h) in birds:
                    cv2.rectangle(
                        frame, 
                        (x, y), 
                        (x+w, y+h), 
                        (0, 200, 0), 
                        2
                    )
                    cv2.putText(
                        frame, 
                        'Detected Bird', 
                        (x, y - 10 if y > 20 else y + 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.6, 
                        (255,0,255), 
                        2
                    )

                # Getting the value of frames per second (FPS)
                cTime = time.time()
                fps = 1/(cTime-self.pTime)
                self.pTime = cTime

                # If detected birds greater than minimum number of birds to be detected
                if (len(birds)>=self.MIN_NUM_BIRDS):
                    self.num_birds=len(birds)

                # Output the value of FPS and Number of birds per frame
                cv2.putText(frame,"Press 'ESC' to exit",(30,70), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 3, cv2.FILLED)
                cv2.putText(frame, "FPS: " + str(int(fps)), (30,120), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 3, cv2.FILLED)
                cv2.putText(frame, f"Birds Detected: {self.num_birds}", (30,170), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 3, cv2.FILLED)


                # Display the resulting frame
                cv2.imshow('Detected Birds', frame)

                # Pressing "ESC" will stop the program
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    self.running = False
            else:
                self.running = False

        # When everything done, release the capture and go back take another one
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n\n\t\t---BIRD DETECTION IN VIDEO---")
    print("\nAUTHORS : \n\t1.Mohamed Asif (2005032)\n\t2.Priyanka S   (2005038)\n\t3.Shrikanth D  (2005046)")
    video = input("\nEnter the video file (in .mp4 format): ")
    D = DetectBirds(video=video)
    D.detect()
    print("\nTHANK YOU! :)")