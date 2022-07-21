import yaml
import os, sys
import socket

def get_host_name():
    return hostname


def get_body_model_path():
    return "extern/models_smplx_v1_1/models"
    # if 'vlg-atlas' in hostname:
    #     bmpath = '/local/home/yanzhang25/body_models/VPoser'
    # elif 'emerald' in hostname:
    #     bmpath = '/home/yzhang/body_models/VPoser'
    # else:
    #     raise ValueError('not stored here')
    # return bmpath

def get_body_marker_path():
    return "extern/models_smplx_v1_1/models/body_markers"
    # if 'vlg-atlas' in hostname:
    #     mkpath = '/local/home/yanzhang25/body_models/Mosh_related'
    # elif 'emerald' in hostname:
    #     mkpath = '/home/yzhang/body_models/Mosh_related'
    # else:
    #     raise ValueError('not stored here')
    # return mkpath

def get_amass_canonicalized_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-data/AMASS-Canonicalized-MP/data'
    elif 'emerald' in hostname:
        mkpath = '/mnt/hdd/datasets/AMASS_SMPLH_G-canon/data'
    else:
        raise ValueError('not stored here')
    return mkpath

def get_amass_canonicalizedx10_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-data/AMASS-Canonicalized-MPx10/data'
    elif 'emerald' in hostname:
        mkpath = '/home/yzhang/Videos/AMASS-Canonicalized-MPx10/data'
    else:
        raise ValueError('not stored here')
    return mkpath




hostname = socket.gethostname()



















