# To json:
# filepath = '/mnt/c/Users/Lukas/Projects/ar-population/Data/GammaResults/MPVAEPolicy_v0/res_000.pkl'
# outpath = '/mnt/c/Users/Lukas/Projects/ar-population/Data/GammaResults/MPVAEPolicy_v0/res_000.json'

# To pickle
# jsonfilepath = '/mnt/c/Users/Lukas/Projects/ar-population/Data/UnityTrajectories/test_traj_1.json'
# pkloutpath = '/mnt/c/Users/Lukas/Projects/ar-population/Data/UnityTrajectories/traj_1_test.pkl'


import pickle
import json
import argparse


import numpy as np

def transform_to_lists(data_dict):
    res = {}
    for key, value in data_dict.items():
        if  type(value) is dict:
            res[key] = transform_to_lists(value)
        elif type(value) is np.ndarray:
            value = np.squeeze(value)
            temp_dict = {"shape": list(value.shape), "data": np.reshape(value, -1).tolist()}
            res[key] = temp_dict
        elif type(value) is list:
            res[key] = [transform_to_lists(x) for x in value]
        elif key == "curr_target_wpath":
            res[key] = {"index": value[0], "position": value[1].tolist()}
        else:
            res[key] = value

    return res


def to_pickle(source_file_path, dest_file_path):
    with open(source_file_path, "rb") as f2:
        res = json.load(f2)
        np_arr = np.asarray(res, dtype=np.float64)
        np_arr = np_arr.reshape(np_arr.size//3, 3)
        with open(dest_file_path, 'wb') as f:
            pickle.dump(np_arr, f)

def to_json(source_file_path, dest_file_path):
    with open(source_file_path, "rb") as f:
        dataall = pickle.load(f, encoding="latin1")
        new_dict = transform_to_lists(dataall)
        json_object = json.dumps(new_dict, indent = 4) 
        with open(dest_file_path, "w") as out:
            out.write(str(json_object))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-toPickle', help='Path to source json file and output pickle file', nargs=2)
    parser.add_argument('-toJson', help='Path to source pickle file and output json file', nargs=2)

    return vars(parser.parse_args())
def main():
        parsed_args = parse_arguments()
        if  (parsed_args['toPickle']):
            to_pickle(parsed_args['toPickle'][0], parsed_args['toPickle'][1])
        elif (parsed_args['toJson']):
            to_json(parsed_args['toJson'][0], parsed_args['toJson'][1])

if __name__ == "__main__":
    main()