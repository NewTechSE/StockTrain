import os

HOST_API = os.environ.get('HOST_API')
HOST_API = HOST_API if HOST_API else "http://localhost:5000/api"