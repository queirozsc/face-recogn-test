import logging
import time
import falcon
import json
import face_recognition

from wsgiref import simple_server
from io import BytesIO
from falcon_multipart.middleware import MultipartMiddleware

known_image = face_recognition.load_image_file("base.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]

logger = logging.getLogger()


def timeit(method):
    """
    Applied as an decorator to get time elapsed.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        data = json.dumps({'Method': method.__name__, 'elapsed':te-ts})
        logger.warning(data)
        return result

    return timed


class ImageDetectionResource(object):
    # @timeit
    def on_post(self, req, resp):
        f = req.get_param('file')
        unknown_encoding = self.load_image(f)
        if unknown_encoding:
            # comp = face_recognition.compare_faces([biden_encoding], unknown_encoding[0])
            comp = self.compare_faces(unknown_encoding)
            if comp:
                resp.body = ("Face Recognized")
            else:
                resp.body = ("Face not Recognized")

        resp.status = falcon.HTTP_200

    @timeit
    def load_image(self, image):
        unknown_image = face_recognition.load_image_file(BytesIO(image.file.read()))
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        return unknown_encoding

    @timeit
    def compare_faces(self, unknown_encoding):
        comp = face_recognition.compare_faces([biden_encoding], unknown_encoding[0])
        return comp

app = falcon.API(middleware=[MultipartMiddleware()])


resources = ImageDetectionResource()
app.add_route('/', resources)

if __name__ == '__main__':
    httpd = simple_server.make_server('127.0.0.1', 8000, app)
    httpd.serve_forever()
