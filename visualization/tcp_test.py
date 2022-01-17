import socket
import time
import json
from datetime import datetime
import uuid
import random

ID = str(uuid.uuid1())[:12]

PORT = 18888
if __name__ == '__main__':
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(('localhost', PORT))
    try:
        while True:
            time.sleep(0.01)
            DUMMY_MSG = json.dumps({"id": ID, 
                                    "timestamp": time.mktime(datetime.utcnow().timetuple()), 
                                    "data": random.randint(0,4096)}) + '\n'
            client_sock.send(bytes(DUMMY_MSG, encoding='ascii'))
    except KeyboardInterrupt:
        client_sock.close()
