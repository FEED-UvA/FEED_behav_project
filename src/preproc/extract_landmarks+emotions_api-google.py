import io
import os
import pickle
import os.path as op
import numpy as np
import pandas as pd

from glob import glob
from google.cloud import vision
from tqdm import tqdm
from joblib import Parallel, delayed


def _run_parallel(directory):

    frames = sorted(glob(directory + '/*/texmap/frame*.png'))
    for frame in frames:
        out = frame.replace('.png', '_api-google_annotations.pkl')
        if not op.isfile(out):
            client = vision.ImageAnnotatorClient()
    
            with io.open(frame, 'rb') as image_file:
                content = image_file.read()
        
            image = vision.types.Image(content=content)    
            response = client.face_detection(image=image)
            with open(out, 'wb') as f_out:
                pickle.dump(response, f_out)
        else:
            pass
            #print(f"Already done {directory}")


if __name__ == '__main__':
    here = op.dirname(__file__)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = op.join(here, '..', '..', "googlevision_key.json")
    ids = sorted(glob('../FEED_stimulus_frames/id*'))
    Parallel(n_jobs=1)(delayed(_run_parallel)(d) for d in tqdm(ids)) 