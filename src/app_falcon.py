import base64
import logging
import time
import falcon
import json
import face_recognition
import os


from wsgiref import simple_server
from io import BytesIO
from falcon_multipart.middleware import MultipartMiddleware


server_port = os.getenv('SERVER_PORT', 5000)

#Preload Base Image.
known_image = face_recognition.load_image_file("base.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]

logger = logging.getLogger()


def timeit(method):
    """
    Applied as an decorator to get time elapsed and generate json logs
    that are compatible with fluentd and elasticsearch.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logs = {'Method': method.__name__, 'elapsed': '{:f}'.format(te-ts)}

        request = [ request for request in args if type(request) == falcon.Request ]
        if request:
            logs.update({'filename': request[0].media.get('filename', 'unknown')})
            logs.update({'client': request[0].media.get('client', 'unknown')})

        data = json.dumps(logs)
        logger.warning(data)
        return result

    return timed


class ImageDetectionResource(object):
    def on_post(self, req, resp):
        """
        Get some info from json request, transform base64 into image
        and make the face_compare.
        """
        identity = req.media.get('identity', False)
        client = req.media.get('client', 'unknown')
        filename = req.media.get('filename', 'unknown_image')

        image = self.base64toimage(identity, req)
        unknown_encoding = self.get_face_encondings(image, req)
        if unknown_encoding:
            comp = self.compare_faces(unknown_encoding, req)
            if comp:
                resp.body = ("Face Recognized")
            else:
                resp.body = ("Face not Recognized")

        resp.status = falcon.HTTP_200

    @timeit
    def base64toimage(self, encoded, req):
        """
        Convert base64string into img.
        """
        base = base64.b64decode(encoded)
        
        return base

    @timeit
    def get_face_encondings(self, image, req):
        unknown_image = face_recognition.load_image_file(BytesIO(image))
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        return unknown_encoding

    @timeit
    def compare_faces(self, unknown_encoding, req):
        comp = face_recognition.compare_faces([biden_encoding], unknown_encoding[0])
        return comp

app = falcon.API(middleware=[MultipartMiddleware()])


resources = ImageDetectionResource()
app.add_route('/', resources)

if __name__ == '__main__':
    httpd = simple_server.make_server('127.0.0.1', server_port, app)
    httpd.serve_forever()
