import tensorflow as tf

MODEL_DIR = '/model/heart_disease_model/1'
print(f'Loading model from {MODEL_DIR}...')
m = tf.saved_model.load(MODEL_DIR)
print('Signatures:', list(m.signatures.keys()))

s = m.signatures['serving_default']
print('Input specs:')
for k, v in s.structured_input_signature[1].items():
    print(f'  {k}: {v}')
print('Output keys:', list(s.structured_outputs.keys()))
print('DONE')
