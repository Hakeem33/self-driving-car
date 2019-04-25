# this file is where we set up our model and the simulator

# we need a socket.io server but before that we need to install flask
import socketio
import eventlet
import numpy as np
from flask import Flask # to create instances of a web application
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# now we initialise our web server
# socket is used to perform real time communication between client and server.
# when client creates a single connection to a web socket sever, it listens for new events
# from server allowing us to continuously update client with data
sio = socketio.Server()

# we can initialise our application by 
app = Flask(__name__) 
speed_limit = 10
# we will use this image to preprocess it just like how we did in our .ipynb file
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    return img

# before making use of model, we need to register a specific event handler which will 
# invoke a function called telemetry which will take in session id and data it receives from simulator
# receives from the simulator
# essentially we are listening for updates that will be sent to telemetry from simulator
# what's happening is, as soon as the connection is established, we're setting the initial
# steering and throttle values and emit to simulator such that the car starts off as stationary and facing forward
# but simulator will send us back data which contains the current image of the frame where the car is locate in track.
# based on the image, we want to run it through our model
# the model will extract features from image and predict the steering angle which we send back to simulation
# and we continue doing this, hence the self driving car
# so first we obtain current image as data['image'] which is base 64 encoded. we must decode it so base64.b64decode(data['image'])
# before we open and identify the given image file with image.open, we need to use a buffer module to mimic our data 
# like a file which we can use for processing. to do this, we make use of bytes.io
# we now use Image.open and assign it to image variable
# we then need to convert our data into an array
# we then set a new image value the involves img_preprocess
# the model now expect 4d arrays whereas our image is just 3d, so we enclose this image inside of another array
# by updating image variable image = np.array([image])
# now we can feed the image into the model we previously loaded so that it predicts an appropriate steering angle. so,
# we set steering_angle = float(model.predict(image))
# we then send the steering angle over to simulation with send_control(steering_angle, 1.0)
# we will print the steering angle, throttlw and spees
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


# when there a connection with the client, we want to fire off an event handler
# to register an event handler we use @sio.on('connect') which fires a connection and will
# invoke the function def connect which takes in our current session id of the client and
# the environment. Upon connection we owuld want to print the string 'Connected'
# as soon as it connects, we will invoke this function send_control and at first we want it to drive straight
# so the steering value will be 0 and we will give a throttle power of 0 to make sure the car
# is initially stationery
# the steering angles themselves will then be determined based on the models predictions
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# now we want to get the car started as soon as the connection is made
# so we define a function called send_control which will take in steering angle and 
# throttle value as arguments
# we want to emit data to simulator. we do this with sio.emit()
# we will then emit a custom event namely steer as this is the event thats going to be 
# listened to by udacity simulator and its going to listen to data that 
# we send in form or key value pairs
# for key 'steering_angle' we send it the steering angle tht we passed in but we will emit it
# as a string value as thats how it will be processed to make the udacity simulation work
# We do the same for throttle
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


# whenever we assign python script, python assigns name to main
# the server will require a middleware to dispatch traffic to socket.io web application with our app, hence our arguments(sio, app)
# we can use a web server gateway interface wsgi to help our server send any request made by client to web application itself
# to launch wsgi server we call eventlet.wsgi.server() .eventlet.listen() opens up listening sockets where we declare Ip and port as atuple.
# second argument would be app where the request is going to be sent
# we will load the model to be used 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
