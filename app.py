
from flask import Flask, request, render_template, jsonify, send_file
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

app = Flask(__name__)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2)
        self.bn_conv1 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=4,
            stride=1,
            padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)
        self.fc1 = nn.Linear(in_features=5 * 5 * 64, out_features=2048)
        self.bn_fc1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.bn_fc2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=7)

    def forward(self, x):
        x = self.bn_x(x)
        x = F.max_pool2d(torch.relu(self.bn_conv1(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         ceil_mode=True)
        x = F.max_pool2d(torch.relu(self.bn_conv2(self.conv2(x))),
                         kernel_size=3,
                         stride=2,
                         ceil_mode=True)
        x = F.max_pool2d(torch.relu(self.bn_conv3(self.conv3(x))),
                         kernel_size=3,
                         stride=2,
                         ceil_mode=True)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EmotionClassifier:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = Model()
        self.model.load_state_dict(
            torch.load('model_params.pkl', map_location='cpu'))
        self.model.eval()

    def get_emotion(self, inputs):
        inputs = self.preprocess(inputs)
        _, predicted = torch.max(self.model(inputs), 1)
        probability = F.softmax((self.model(inputs)), dim=1).detach().numpy().flatten()
        emotion = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprised',
            6: 'normal'
        }
        return emotion[predicted.item()], probability

    def preprocess(self, inputs):
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        inputs = trans(inputs)
        inputs = inputs.unsqueeze(0)
        return inputs

emo_cls = EmotionClassifier()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img = Image.open(file.stream)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = emo_cls.face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:(y + h), x:(x + w)], (42, 42))
            emotion, probability = emo_cls.get_emotion(Image.fromarray(face))
            results = {
                'emotion': emotion,
                'probability': probability.tolist()
            }
            return jsonify(results)
    return jsonify({'error': 'No face detected'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
