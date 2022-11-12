# CNN
## Create experiment
For simple convolution neural network (without residual, dense connections) You can specify architecture in yaml file.  
Example of yaml file:
```yaml
version: 1

defaults:
  experiment_name: "baseline"
  epochs: 5
  batch_size: 64

optimizer:
  name: "Adam"
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999

architecture:
  convolution_layers:
    conv_layer_1:
      type: "Conv2d"
      in_channels: 3
      out_channels: 32
      kernel_size: (3,3)
      stride: (1,1)
      padding: (1,1)
    pooling_1:
      type: "MaxPool2d"
      kernel_size: (2,2)
      stride: (1,1)
    activation_function_1:
      type: "ReLU"
    conv_layer_2:
      type: "Conv2d"
      in_channels: 32
      out_channels: 64
      kernel_size: (3,3)
      stride: (1,1)
      padding: (1,1)
    pooling_2:
      type: "MaxPool2d"
      kernel_size: (2,2)
      stride: (1,1)
    activation_function_2:
      type: "ReLU"
    conv_layer_3:
      type: "Conv2d"
      in_channels: 64
      out_channels: 128
      kernel_size: (3,3)
      stride: (1,1)
      padding: (1,1)
    pooling_3:
      type: "MaxPool2d"
      kernel_size: (2,2)
    activation_function_3:
      type: "ReLU"
  fully_connected_layers:
    fc_layer_1:
      type: "Linear"
      in_features: 2048
      out_features: 10

```
When yaml is finished You just need to run:
```bash
python main.py --config my_yaml_file.yaml
```
For more complex architectures you can create class which name need to be `CustomCnnModelCifar10` where You will specify architecture. When class is created with  architecture You need to create yaml file.
```yaml
version: 1

defaults:
  experiment_name: "deepmodel"
  epochs: 20
  batch_size: 64

optimizer:
  name: "Adam"
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999

architecture:
  class: "CustomCnnModelCifar10"
```
Where you will specify class.
