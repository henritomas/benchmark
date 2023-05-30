import numpy
import onnx

import os
import torch
import onnxruntime as rt
import time
import numpy as np
import timm
import torchvision

from argparse import ArgumentParser

# ARGUMENTS
parser = ArgumentParser(description='EfficientDL')
parser.add_argument('-t', '--torch',  action="store_true", help='switch if no onnx optimization is applied, torch model only.')
parser.add_argument('-o', '--optim', choices=['disable', 'basic', 'extended', 'all'], default='all')
args = parser.parse_args()

# SETTINGS
USE_GPU = False # 'CUDAExecutionProvider'
USE_TORCH_MODEL = args.torch
WARM_UP_REPS = 1000
REPETITIONS = 1000
OPTIM_LEVEL = args.optim

# Determine Optimization level to use
graph_optim_mapper = {
    "disable": rt.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": rt.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": rt.GraphOptimizationLevel.ORT_ENABLE_ALL
}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def change_ir_version(filename, ir_version=6):
    "onnxruntime==1.2.0 does not support opset <= 7 and ir_version > 6"
    with open(filename, "rb") as f:
        model = onnx.load(f)
    model.ir_version = 6
    if model.opset_import[0].version <= 7:
        model.opset_import[0].version = 11
    return model

def get_torchvision_models():
    models = list(torchvision.models.__dict__.keys())
    torchvision_models = []
    for model in models:
        if model.islower() and "__" not in model and model[0] != "_":
            torchvision_models.append(model)

    return torchvision_models

def get_timm_models():
    timm_models = timm.list_models(pretrained=True)
    return timm_models

def download_model(model_name):
    timm_models = get_timm_models()
    torchvision_models = get_torchvision_models()

    if model_name in timm_models:
        print(f"Using timm model: {model_name}")
        model = timm.create_model(model_name, pretrained=True, exportable=True).to("cpu")
    elif model_name in torchvision_models:
        print(f"Using torchvision model: {model_name}")
        model_name = "torchvision.models." + model_name
        model = eval(model_name)(pretrained=True).to("cpu")
    else:
        print("Model not found in torchvision or timm", model_name)
        exit(0)

    return model

# CPU OR GPU
provider = 'CUDAExecutionProvider' if USE_GPU else 'CPUExecutionProvider'

models = [
    "regnety_002",
    "regnety_004",
    "regnety_006",
    "regnety_008",
    # "regnetx_016",
]

for model_id in models:

    MODEL_PATH = f"models/model_{model_id}.onnx"

    model = download_model(model_id)
    onnx_model = MODEL_PATH

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float,).to("cpu")
    with torch.no_grad():
        torch.onnx.export(model,              
                            dummy_input,
                            onnx_model,
                            # store the trained parameter weights inside the model file
                            export_params=True,
                            # whether to execute constant folding for optimization   
                            do_constant_folding=True,  
                            input_names=['inputs'],
                            output_names=['outputs'],
                            verbose=False
                            )

    onnx_model = change_ir_version(MODEL_PATH)
    onnx_model_str = onnx_model.SerializeToString()

    # WARM-UP
    options = rt.SessionOptions()
    # graph optimization
    options.graph_optimization_level = graph_optim_mapper[OPTIM_LEVEL]
    sess_profile = rt.InferenceSession(onnx_model_str, options, providers=[provider])
    dummy_input_torch = torch.randn(1, 3, 224, 224, dtype=torch.float,).to("cpu")
    dummy_input = {sess_profile.get_inputs()[0].name: to_numpy(dummy_input_torch)}

    if USE_TORCH_MODEL:
        model.eval()
        with torch.no_grad():
            for _ in range(WARM_UP_REPS):
                _ = model(dummy_input_torch)
    else:
        for _ in range(WARM_UP_REPS):
            _ = sess_profile.run(None, dummy_input)

    # WITHOUT PROFILER
    timings = []
    if USE_TORCH_MODEL:
        model.eval()
        with torch.no_grad():
            for rep in range(REPETITIONS):
                start = time.time()
                y = model(dummy_input_torch)
                y = y.detach().cpu().numpy()
                timings.append((time.time() - start)  * 1e6)
    else: 
        for rep in range(REPETITIONS):
            start = time.time()
            _ = sess_profile.run(None, dummy_input)
            timings.append((time.time() - start)  * 1e6)

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    print('=' * 50)
    print(model_id.upper())
    
    print("(CPU Timer) Ave infer time: {0:.1f} usec".format(mean_syn))
    print("(CPU Timer) Std infer time: {0:.1f} usec".format(std_syn))

#     # PROFILING
#     options = rt.SessionOptions()
#     # graph optimization
#     options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
#     options.enable_profiling = True
#     sess_profile = rt.InferenceSession(onnx_model_str, options, providers=[provider])

#     #res = sess_profile.run(None, dummy_input)
#     timings = []
#     for rep in range(REPETITIONS):
#         start = time.time()
#         _ = sess_profile.run(None, dummy_input)
#         timings.append((time.time() - start)  * 1e6)

#     prof_file = sess_profile.end_profiling()
#     os.rename(prof_file, f"onnx_traces/{model}_onnx_trace.json") 

#     mean_syn = np.mean(timings)
#     std_syn = np.std(timings)

#     print("(PROFILE TIMER) Ave infer time: {0:.1f} usec".format(mean_syn))
#     print("(PROFILE TIMER) Std infer time: {0:.1f} usec".format(std_syn))
#     print("PROFILE saved at: ", prof_file)
    