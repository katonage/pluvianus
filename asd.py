
import faulthandler
faulthandler.enable()
import caiman
#import cpuinfo
import platform, os
print("Python:", platform.python_version())
#print("CPU flags:", " ".join(sorted(cpuinfo.get_cpu_info().get("flags", []))))
print("TF_ENABLE_ONEDNN_OPTS =", os.environ.get("TF_ENABLE_ONEDNN_OPTS"))
try:
    import tensorflow as tf
    from tensorflow.python.platform import build_info as bi
    print("TF:", tf.__version__)
    print("is_cuda_build:", bi.build_info.get("is_cuda_build"))
except Exception as e:
    print("TF import error:", e)
    
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from tensorflow import keras
import numpy as np, json

json_path = r"C:\Users\KatonaGergely\caiman_data\model\cnn_model.json"
weights_path = r"C:\Users\KatonaGergely\caiman_data\model\cnn_model.h5"  # adjust if needed

with open(json_path, "r", encoding="utf-8") as f:
    model = keras.models.model_from_json(f.read())
model.load_weights(weights_path)

print("Model input shape:", model.input_shape)  # e.g., (None, 50, 50, 1)

H, W, C = model.input_shape[1:]  # (50, 50, 1)
x = np.random.rand(5, H, W, C).astype("float32")
y = model.predict(x, verbose=1)
print("Predict OK. y.shape:", y.shape)