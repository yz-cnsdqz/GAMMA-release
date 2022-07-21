# Python 3 server example
# http://127.0.0.1:8080/
# curl --data "[-1.689821, -1.027213, -1.733333, -1.733333, -1.027213, -1.666667, -2.066667, -1.010546, -1.666667, -2.4, -1.010546, -1.766666, -2.833333, -1.010546, -2]" --header "Content-Type: application/json" http://localhost:8080
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
import os
import numpy as np
import pickle
import subprocess

hostName = "localhost"
serverPort = 8080

class GammaServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(bytes(json.dumps({'hello': 'world', 'received': 'ok'}),"utf-8"))

    def do_POST(self):
        # refuse to receive non-json content
        if self.headers['Content-Type'] != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        length = int(self.headers['Content-Length'])
        message = json.loads(self.rfile.read(length))
        self.handle_request(message)
        
        # send the message back
        self._set_headers()
        self.wfile.write(bytes(json.dumps(message), "utf-8"))
    
    def handle_request(self, message):
        if not os.path.exists("TempData"):
            os.mkdir("TempData")
        np_arr = np.asarray(message, dtype=np.float64)
        np_arr = np_arr.reshape(np_arr.size//3, 3)
        with open("TempData/traj_1.pkl", 'wb') as f:
            pickle.dump(np_arr, f)

        out = subprocess.check_output("cd /mnt/c/Users/Lukas/Projects/GAMMA-release/ && /home/lukas/miniconda3/envs/gamma/bin/python exp_GAMMAPrimitive/gen_motion_long_in_Cubes.py --cfg MPVAEPolicy_v0", shell=True)
        # print(out)

        


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), GammaServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")