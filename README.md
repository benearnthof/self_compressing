# self_compressing
Implementation of Self-Compressing Neural Networks in Pytorch https://arxiv.org/abs/2301.13142

Will add nice trainer class if there is demand, for now trainer.py trains with batch size of 512 for 20k steps.  
5k steps yield 97% accuracy at 22000 bytes model size, the full 20k steps yield 98.5% accuracy at 18800 bytes. Training performed on a single A100 40GB.  
Training was done without hyperparameter tuning, for models that are more complex I'd recommend adjusting the Quantization weight in the loss calculation in `trainer.train_step`.

Implementation only includes a quantized 2d convolutional layer, other layers would be implemented with quantization analogous to `QConv2d.qweight`.