Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteTaskVision'
  s.version          = '0.1.6'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache',:file => "LICENSE"}
  s.homepage         = 'https://github.com/tensorflow/tflite-support'
  s.source           = { :http => 'https://dl.dropboxusercontent.com/s/47aapsp0a00i1q4/TensorFlowLiteTaskVisionCoreml-0.1.6-dev.tar.gz?dl=0' }
  s.summary          = 'TensorFlow Lite Task Library - Vision'
  s.description      = 'The Computer Vision APIs of the TFLite Task Library'

  s.ios.deployment_target = '11.0'

  s.module_name = 'TensorFlowLiteTaskVision'
  s.static_framework = true
  s.pod_target_xcconfig = {
     'VALID_ARCHS' => 'x86_64, arm64, armv7',
  }
  s.library = 'c++'
  s.frameworks = 'CoreMedia', 'Accelerate', 'CoreML'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteTaskVisionCoreml_framework.framework'
end
