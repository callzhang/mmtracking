import enum
import cv2, torch, requests, json, numpy, os
from numpy.core.fromnumeric import shape
from cv2 import data
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from mmtrack.apis import inference_sot, init_model
from box import Box
from io import BytesIO
import numpy as np

def main():
    args = Box()
    args.config = 'configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py'
    args.input = 'demo/video_10s.mp4'
    args.output = 'output/sot.mp4'
    args.checkpoint = 'models/siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.show = True
    args.color = (0, 255, 0)
    args.thickness = 3

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    cap = cv2.VideoCapture(args.input)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # get first frame detection
    flag, frame = cap.read()
    prediction = get_first_detection(frame)
    all_tracks = []
    for frame, detection in enumerate(prediction):
        class_ = detection['label']
        conf = detection['confidence']
        xyxy = detection['xyxy']
        tracks = []
        for frame_id in tqdm(enumerate(frames)):
            img = cap.get('cv.CAP_PROP_POS_FRAME')
            track = inference_sot(model, img, xyxy, frame_id)
            track_bbox = track['bbox']
            score = track['score']
            track['frame'] = frame_id
            tracks.append(track)
            # color = np.random.randint(1, 255, size=(3))
        all_tracks.append(tracks)


def get_first_detection(img):
    tmp_img = 'temp/temp_img.jpg'
    os.makedirs('temp', exist_ok=True)
    Image.fromarray(img).save(tmp_img)
    res = requests.post('http://localhost:5005/detection',
        data={
            'view_img': 0
        },
        files={
            'image': ('first_frame.jpg', open(tmp_img, 'rb'), 'image/jpg')
        }
    )
    if res.status_code == 200:
        if type(res.content) is bytes:
            Image.open(BytesIO(res.content)).save('temp/result.jpg')
            print('image saved to temp/result.jpg')
        else:
            #parse result
            result = res.json()
            desc = result['description']
            print(desc)
            prediction = result['prediction']
            return prediction
                
def sot(frame_id, class_, xyxy):
    result = inference_sot(model, frame, class_, xyxy, frame_id)


if __name__ == '__main__':
    args, model, cap = init()
    flag, frame = cap.read()
    prediction = get_first_detection(frame)
    # parse and send
    for frame, detection in enumerate(prediction):
        class_ = detection['label']
        conf = detection['confidence']
        xyxy = detection['xyxy']
        tracks = sot(frame, class_, xyxy)


