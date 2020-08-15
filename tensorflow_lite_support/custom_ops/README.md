# Custom Ops for Tensorflow Text API

This package provides Tensorflow Lite custom ops and utility wrappers to make
TFLite models trained with Tensorflow Text can be convertibel and executed
efficiently.

Current we have:
* TFLite friendly Sentencepiece Tokenizer
* Fused Whitespace Tokenizer in TFLite
* Fused ngram in TFLite
* TFLite inference with flex delegate.
