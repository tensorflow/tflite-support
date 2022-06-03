Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteTaskVisionObjCPP'
  s.version          = '0.1.1'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache',:file => "LICENSE"}
  s.homepage         = 'https://github.com/tensorflow/tflite-support'
  s.source           = { :http => 'https://dl.dropboxusercontent.com/s/5v2775z785b8114/TensorFlowLiteTaskVisionObjCPP-0.1.1-dev.tar.gz?dl=0' }
  s.summary          = 'TensorFlow Lite Task Library - Vision'
  s.description      = 'The Computer Vision APIs of the TFLite Task Library'

  s.ios.deployment_target = '10.0'

  s.module_name = 'TensorFlowLiteTaskVisionObjCPP'
  s.static_framework = true
  s.pod_target_xcconfig = {
     'VALID_ARCHS' => 'x86_64, arm64, armv7',
  }
  s.library = 'c++'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteTaskVisionObjCPP.framework'
end
