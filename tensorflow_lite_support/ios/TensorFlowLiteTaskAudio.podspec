Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteTaskAudio'
  s.version          = '0.0.5'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache',:file => "LICENSE"}
  # https://www.dropbox.com/s/1vkwr3g6scmm5dx/TensorFlowLiteTaskAudio-0.0.5-dev.tar.gz?dl=0
  s.homepage         = 'https://github.com/tensorflow/tflite-support'
  s.source           = { :http => 'https://dl.dropboxusercontent.com/s/1vkwr3g6scmm5dx/TensorFlowLiteTaskAudio-0.0.5-dev.tar.gz?dl=0' }
  s.summary          = 'TensorFlow Lite Task Library - Audio'
  s.description      = 'The Audio APIs of the TFLite Task Library'

  s.ios.deployment_target = '10.0'

  s.module_name = 'TensorFlowLiteTaskAudio'
  s.static_framework = true
  s.pod_target_xcconfig = {
     'VALID_ARCHS' => 'x86_64, arm64, armv7',
  }
  s.library = 'c++'
  s.frameworks = 'AVFoundation'
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteTaskAudio.framework'
end