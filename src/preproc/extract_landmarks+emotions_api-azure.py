import io
import json
import pickle
import os.path as op
from glob import glob
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from joblib import Parallel, delayed
from msrest.authentication import CognitiveServicesCredentials
from tqdm import tqdm


def _run_parallel(face_client, directory):

    frames = sorted(glob(directory + '/*/texmap/frame*.png'))
    for frame in frames:
        out = frame.replace('.png', '_api-azure_annotations.pkl')
        if not op.isfile(out):
            response = face_client.face.detect_with_stream(
                image=open(frame, 'rb'),
                return_face_attributes=[FaceAttributeType.emotion],
                return_face_landmarks=True,
                return_face_id=False
            )
            with open(out, 'wb') as f_out:
                pickle.dump(response, f_out)


if __name__ == '__main__':
    here = op.dirname(__file__)
    with open(op.join(here, '..', '..', 'azure_key.json'), 'rb') as f_in:
        azure_info = json.load(f_in)

    face_client = FaceClient(
        azure_info['ENDPOINT'],
        CognitiveServicesCredentials(azure_info['KEY'])
    )

    ids = sorted(glob('../FEED_stimulus_frames/id*'))
    Parallel(n_jobs=1)(delayed(_run_parallel)(face_client, d) for d in tqdm(ids))
