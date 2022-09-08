# Python 3 server example
# http://127.0.0.1:8080/
# curl --data "[-1.689821, -1.027213, -1.733333, -1.733333, -1.027213, -1.666667, -2.066667, -1.010546, -1.666667, -2.4, -1.010546, -1.766666, -2.833333, -1.010546, -2]" --header "Content-Type: application/json" http://localhost:8080
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import numpy as np
import pickle
from exp_GAMMAPrimitive.gen_motion_long_in_Cubes import run
import socket   
import random

np.random.seed(42)
N_MALE_BETA_SHAPES = 3
N_FEMALE_BETA_SHAPES = 3
female_betas = np.random.randn(N_FEMALE_BETA_SHAPES, 1, 10)
male_betas =  np.random.randn(N_MALE_BETA_SHAPES, 1, 10)

# female_betas = np.asarray([
#     [[-1.135557, 0.4900467, -1.571586, 0.324353, 0.6154498, 1.360919, -0.4199827, 0.68612, -0.5775192, 0.09844136]], 
#     [[-3.77, 0.07, 1.188608, 3.63, -4.37, 2.73, -3.92, -1.99, 5, 1.536026]],
#     [[-0.05453706, 5, 0.3141453, -1.396356, -0.2505138, 0.9503168, 1.143413, -1.807971, -0.9851396, -1.033214]]], dtype=np.float32)
# male_betas =np.asarray([
#     [[-0.9842331, -0.9684765, -1.111603, 1.859693, -1.533256, 0.8656253, -0.542197, -0.4216726, 0.1279395, 0.4362581]],
#     [[-2.02, 1.44, 0.6797122, -1.945877, -0.4878228, 1.552606, -0.9787962, 0.04247403, 1.987148, 0.4292719]],
#     [[1.652401, 3.57, 0.261369, -1.349104, 0.858787, -1.169071, 0.4518526, -1.11841, -0.1033189, 1.916406]]], dtype=np.float32)
# female_betas = np.zeros((1, 1, 10))
# male_betas = np.zeros((1, 1, 10))

# print("female betas: ")
# print(female_betas)
# print("male betas: ")
# print(male_betas)

gammaResultsDir = "GammaResults"
gammaSourceDir = "GammaSource"
gammaResultFileName = "results.pkl"
STORE_JSON = True
gammaResultsJsonFileName = "results.json"
IGNORED_KEYS = ["markers", "markers_proj", "joints", "mp_latent", "timestamp", "curr_target_wpath"]

# hostName = "localhost"
# hostName = "172.24.85.77"
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
        m_str = self.rfile.read(length)
        print("message: "+ m_str.decode("utf-8") )
        message = json.loads(m_str)
        self.delete_logs()

        response = self.handle_request(message)
        
        # send the message back
        self._set_headers()
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
    
    def handle_request(self, message):
        np_arr = np.asarray(message, dtype=np.float64)
        np_arr = np_arr.reshape(np_arr.size//3, 3)
        with open(os.path.join(gammaSourceDir, "traj_1.pkl"), 'wb') as f:
            pickle.dump(np_arr, f)
        
        gender = random.choice(['female', 'male'])
        if gender == 'female':
            betas =  random.choice(female_betas)
        else:
            betas =  random.choice(male_betas)
        print("Betas: ", betas)
        
        args = {"cfg_policy": 'Gamma_policy_guggenheim_v5',
                'max_depth': 120, 
                'ground_euler': [0, 0, 0], 
                'gpu_index': -1, 
                'random_seed': None, 
                'verbose': 1,
                'gender': gender,
                'betas': betas}
        run(args)
        json_results = self.to_json(os.path.join(gammaResultsDir, gammaResultFileName))
        if STORE_JSON:
            with open(os.path.join(gammaResultsDir, gammaResultsJsonFileName), "w") as out:
                out.write(str(json_results))
        return json_results

        # out = subprocess.check_output("conda init bash && conda activate gamma && python exp_GAMMAPrimitive/gen_motion_long_in_Cubes.py --cfg MPVAEPolicy_v0", shell=True)
        # print(out)
    
    def to_json(self, source_file_path):
        with open(source_file_path, "rb") as f:
            dataall = pickle.load(f, encoding="latin1")
            new_dict = self.transform_to_lists(dataall)
            json_object = json.dumps(new_dict, indent = 4)
            return json_object
    
    def transform_to_lists(self, data_dict):
        res = {}
        for key, value in data_dict.items():
            if key in IGNORED_KEYS:
                continue
            elif  type(value) is dict:
                res[key] = self.transform_to_lists(value)
            elif type(value) is np.ndarray:
                value = np.squeeze(value)
                temp_dict = {"shape": list(value.shape), "data": np.reshape(value, -1).tolist()}
                res[key] = temp_dict
            elif type(value) is list:
                res[key] = [self.transform_to_lists(x) for x in value]
            elif key == "curr_target_wpath":
                res[key] = {"index": value[0], "position": value[1].tolist()}
            else:
                res[key] = value

        return res

    def delete_logs(self):
        for item in os.listdir(gammaResultsDir):
            if item.endswith(".log"):
                os.remove(os.path.join(gammaResultsDir, item))
    
    
def get_ip_address():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

if __name__ == "__main__": 
    ip_address = get_ip_address()
    webServer = HTTPServer((ip_address, serverPort), GammaServer)
    print("Server started http://%s:%s" % (ip_address, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")