import tensorflow as tf
import os

# �]�m TensorFlow ��x�ŧO�� DEBUG�A�H�K�ݨ��h�ԲӫH��
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # '0' for all logs, '3' for no logs

print("TensorFlow Version:", tf.__version__)

# �ˬd�O�_������P CUDA �� cuDNN �����������ܼơA�H�T�O���̨S���Ĭ�
print("\nCUDA_PATH:", os.getenv('CUDA_PATH'))
print("CUDNN_PATH:", os.getenv('CUDNN_PATH')) # �p�G�z��ʳ]�m�L�o��

print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and detected!")
    print("GPU Details:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  Name: {gpu.name}, Type: {gpu.device_type}")

    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Simple matrix multiplication on GPU successful.")
        print(c)
    except Exception as e:
        print(f"Error during GPU test operation: {e}")
else:
    print("No GPU detected. Please re-check your NVIDIA driver, CUDA Toolkit, and cuDNN installations.")

print("\nLogical Devices:")
print(tf.config.list_logical_devices())

print("\nAvailable Physical Devices (including CPU):")
print(tf.config.list_physical_devices())