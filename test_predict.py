import struct, base64, requests, json

def varint(n):
    buf = []
    while True:
        bits = n & 0x7F
        n >>= 7
        if n:
            buf.append(0x80 | bits)
        else:
            buf.append(bits)
            break
    return bytes(buf)

def make_bytes_feature(name, float_vals=None, int64_vals=None):
    name_bytes = name.encode('utf-8')
    if float_vals is not None:
        floats = b''.join(struct.pack('<f', v) for v in float_vals)
        float_list = b'\x0a' + varint(len(floats)) + floats
        feature_val = b'\x0a' + varint(len(float_list)) + float_list
    else:
        int_bytes = b''.join(varint(v) for v in int64_vals)
        int_list = b'\x0a' + varint(len(int_bytes)) + int_bytes
        feature_val = b'\x1a' + varint(len(int_list)) + int_list
    key_field = b'\x0a' + varint(len(name_bytes)) + name_bytes
    val_field = b'\x12' + varint(len(feature_val)) + feature_val
    map_entry = key_field + val_field
    return b'\x0a' + varint(len(map_entry)) + map_entry

# Pasien dengan risiko jantung (target=1): age=63, sex=1, cp=3...
features = b''
features += make_bytes_feature('age',      float_vals=[63.0])
features += make_bytes_feature('trestbps', float_vals=[145.0])
features += make_bytes_feature('chol',     float_vals=[233.0])
features += make_bytes_feature('thalach',  float_vals=[150.0])
features += make_bytes_feature('oldpeak',  float_vals=[2.3])
features += make_bytes_feature('ca',       float_vals=[0.0])
features += make_bytes_feature('sex',      int64_vals=[1])
features += make_bytes_feature('cp',       int64_vals=[3])
features += make_bytes_feature('fbs',      int64_vals=[1])
features += make_bytes_feature('restecg',  int64_vals=[0])
features += make_bytes_feature('exang',    int64_vals=[0])
features += make_bytes_feature('slope',    int64_vals=[0])
features += make_bytes_feature('thal',     int64_vals=[1])

features_msg = b'\x0a' + varint(len(features)) + features
b64_str = base64.b64encode(features_msg).decode('utf-8')

payload = json.dumps({'instances': [{'examples': {'b64': b64_str}}]})
resp = requests.post(
    'https://heart-disease-mlops-production.up.railway.app/v1/models/heart_disease_model:predict',
    data=payload,
    headers={'Content-Type': 'application/json'},
    timeout=15
)
print(f'Status: {resp.status_code}')
print(resp.text[:300])
