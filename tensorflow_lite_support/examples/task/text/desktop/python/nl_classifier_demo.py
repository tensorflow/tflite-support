import subprocess
import argparse
import os

def classify(model_path, text):
    """Classifies input text into different categories

    Initially the function will download the model and then runs the detection tool.

    Args:
        model_path: path to model
        text: input text

    Returns:
        void
        
    """
    # Download the model:
    if not(os.path.isfile(model_path)):
        subprocess.run(["curl -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite' -o " + model_path + ""], shell=True)

    # Run the detection tool:
    subprocess.run(["bazel run -c opt  tensorflow_lite_support/examples/task/text/desktop:nl_classifier_demo --  --model_path=" + model_path + "  --text='" + text + "'"], shell=True)


def main():
    # Create the parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    # Call the classify method
    classify(args.model_path, args.text)

if __name__ == '__main__':
    main()


    






