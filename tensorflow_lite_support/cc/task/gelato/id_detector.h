
#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_GELATO_ID_DETECTOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_GELATO_ID_DETECTOR_H_
namespace tflite::task::gelato {
  class IdDetector : BaseTaskApi {

// preprocess - normalize to -1 to 1, no grayscale
//    export function prepareImageForModel(
//      image: Tensor,
//      mobilenetScaling: ?boolean,
//    grayscale: ?boolean = false,
//    ) {
//      if (!RGB_COEF) {
//        RGB_COEF = tf.tensor1d([0.2989, 0.587, 0.114]);
//      }
//      return tf.tidy(() => {
//        let img = image;
//        if (grayscale) {
//          img = tf.sum(image.mul(RGB_COEF), 2).expandDims(-1);
//        }
//        // Expand the outer most dimension so we have a batch size of 1.
//        const batchedImage = img.expandDims(0);
//
//        // Normalize the image between -1 and 1. The image comes in between 0-255,
//        // so we divide by 127.5 and subtract 1.
//        // This matches the preprocessing we do in keras
//        // https://github.com/keras-team/keras-applications/blob/43ac53e491fab09b9d938dadeee1e82c56d5d25c/keras_applications/imagenet_utils.py#L121
//        if (mobilenetScaling) {
//          return batchedImage
//            .toFloat()
//            .div(tf.scalar(127.5))
//            .sub(tf.scalar(1));
//        }
//        return batchedImage.toFloat();
//      });
//    }

  };
}


#endif //TENSORFLOW_LITE_SUPPORT_CC_TASK_GELATO_ID_DETECTOR_H_
