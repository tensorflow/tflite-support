Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteTaskVision'
  s.version          = '5.0.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache',:file => "LICENSE" }
  #https://www.dropbox.com/s/vns5malr1sxy8u4/TensorFlowLiteTaskVision-5.0.0-dev.tar.gz?dl=0
  s.homepage         = 'https://github.com/tensorflow/tflite-support'
  s.source           = { :http => 'https://dl.dropboxusercontent.com/s/vns5malr1sxy8u4/TensorFlowLiteTaskVision-5.0.0-dev.tar.gz?dl=0' }
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
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteTaskVision.framework'
end