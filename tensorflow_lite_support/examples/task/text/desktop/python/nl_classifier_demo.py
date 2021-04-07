import ctypes
import time
import argparse

def Classify(model_path, text, input_tensor_name, output_score_tensor_name):
    """Classifies input text into different categories

    Initially the function will import shared library (.so) of the 
    C++ API (cc/invoke_nl_classifier.cc). Then it will initialize 
    the classifier and run the inference using the shared library.

    Args:
        model_path: path to model
        text: input text
        input_tensor_name: input tensor name
        output_score_tensor_name: output tensor name

    Returns:
        void
        
    Raises:
        OSError: An error occurred accessing the shared library. Most probably the file is not found
    """
    # Import shared libary
    # User needs to run `bazel build -c opt \ tensorflow_lite_support/examples/task/text/desktop/python/cc:invoke_nl_classifier`
    # to create the library
    shared_library = ctypes.CDLL('bazel-bin/tensorflow_lite_support/examples/task/text/desktop/python/cc/libinvoke_nl_classifier.so')
    # Initialize a classifier
    shared_library.InvokeInitializeModel.restype = None
    shared_library.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 4)(bytes(model_path, encoding='utf-8'), bytes(input_tensor_name, encoding='utf-8'), bytes(output_score_tensor_name, encoding='utf-8'))
    shared_library.InvokeInitializeModel(len(args),args)

    # Run the inference
    shared_library.InvokeRunInference.restype = ctypes.c_char_p
    shared_library.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    string_buffers = [ctypes.create_string_buffer(8) for i in range(4)]
    pointers = (ctypes.c_char_p*4)(*map(ctypes.addressof, string_buffers))
    args = (ctypes.c_char_p * 1)(bytes(text, encoding='utf-8'))
    shared_library.InvokeRunInference(len(args), args, pointers)
    results = [(s.value).decode('utf-8') for s in string_buffers]
    
    # Print the output
    for i in range(int(len(results)/2)): 
        print("category[{}]: '{}' : '{}'".format(i, results[i*2], round(float(results[(i*2)+1]), 5)))


def main():
    # Create the parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--input_tensor_name', required=True)
    parser.add_argument('--output_score_tensor_name', required=True)
    args = parser.parse_args()
    # Call the Classify method
    Classify(args.model_path, args.text, args.input_tensor_name, args.output_score_tensor_name)

if __name__ == '__main__':
    main()