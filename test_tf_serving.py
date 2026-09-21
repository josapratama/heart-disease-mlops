import requests, json, base64, struct

# Check metadata
r = requests.get('https://heart-disease-mlops-production.up.railway.app/v1/models/heart_disease_model/metadata')
d = r.json()
sd = d['metadata']['signature_def']['signature_def']['serving_default']
print('serving_default inputs:', list(sd['inputs'].keys()))
print('serving_default method:', sd['method_name'])
print()

# Build tf.Example manually using protobuf encoding without TF library
# Wire types: 0=varint, 2=length-delimited
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

def encode_float_feature(values):
    # FloatList { value: float } -> field 1, wire 2
    floats = b''.join(struct.pack('<f', v) for v in values)
    float_list = b'\x0a' + varint(len(floats)) + floats   # field 1 = value
    feature = b'\x0a' + varint(len(float_list)) + float_list  # field 1 = float_list
    return feature

def encode_int64_feature(values):
    # Int64List { value: int64 }
    ints = b''.join(varint(v) for v in values)
    int_list = b'\x0a' + varint(len(ints)) + ints   # field 1 = value (wire 0, varint)
    # Actually Int64List uses field 1 with repeated int64
    # Re-encode properly
    int_bytes = b''.join(varint(v) for v in values)
    # field 1 tag = 0x0a (field 1, wire 2 = length delimited)
    int_list_msg = b'\x0a' + varint(len(int_bytes)) + int_bytes
    feature = b'\x1a' + varint(len(int_list_msg)) + int_list_msg  # field 3 = int64_list
    return feature

# Better: use struct-based approach matching proto3
# tf.Example -> Features -> map<string, Feature>
# Let's use a simpler approach: encode using the known proto layout

def make_bytes_feature(name, float_vals=None, int64_vals=None):
    """Returns (name_bytes, feature_bytes) for a single feature"""
    name_bytes = name.encode('utf-8')
    if float_vals is not None:
        # FloatList
        floats = b''.join(struct.pack('<f', v) for v in float_vals)
        float_list = b'\x0a' + varint(len(floats)) + floats
        feature_val = b'\x02' + varint(len(float_list)) + float_list  # field 1, wire 2
        # Actually Feature field 1 = float_list, tag = 0x0a
        feature_val = b'\x0a' + varint(len(float_list)) + float_list
    else:
        # Int64List - field 3 in Feature
        int_bytes = b''
        for v in int64_vals:
            int_bytes += varint(v)
        int_list = b'\x0a' + varint(len(int_bytes)) + int_bytes
        feature_val = b'\x1a' + varint(len(int_list)) + int_list

    # MapEntry: field 1 = key (string), field 2 = value (Feature message)
    key_field = b'\x0a' + varint(len(name_bytes)) + name_bytes
    val_field = b'\x12' + varint(len(feature_val)) + feature_val
    map_entry = key_field + val_field
    # Features.feature field is 1, wire 2
    feature_map_field = b'\x0a' + varint(len(map_entry)) + map_entry
    return feature_map_field

# Patient: age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1, restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1
features_bytes = b''
features_bytes += make_bytes_feature('age',      float_vals=[63.0])
features_bytes += make_bytes_feature('trestbps', float_vals=[145.0])
features_bytes += make_bytes_feature('chol',     float_vals=[233.0])
features_bytes += make_bytes_feature('thalach',  float_vals=[150.0])
features_bytes += make_bytes_feature('oldpeak',  float_vals=[2.3])
features_bytes += make_bytes_feature('ca',       float_vals=[0.0])
features_bytes += make_bytes_feature('sex',      int64_vals=[1])
features_bytes += make_bytes_feature('cp',       int64_vals=[3])
features_bytes += make_bytes_feature('fbs',      int64_vals=[1])
features_bytes += make_bytes_feature('restecg',  int64_vals=[0])
features_bytes += make_bytes_feature('exang',    int64_vals=[0])
features_bytes += make_bytes_feature('slope',    int64_vals=[0])
features_bytes += make_bytes_feature('thal',     int64_vals=[1])

# Example.features field 1
features_msg = b'\x0a' + varint(len(features_bytes)) + features_bytes
b64_str = base64.b64encode(features_msg).decode('utf-8')

payload = json.dumps({'instances': [{'examples': {'b64': b64_str}}]})
resp = requests.post(
    'https://heart-disease-mlops-production.up.railway.app/v1/models/heart_disease_model:predict',
    data=payload,
    headers={'Content-Type': 'application/json'},
    timeout=15
)
print(f'Status: {resp.status_code}')
print(resp.text[:500])
