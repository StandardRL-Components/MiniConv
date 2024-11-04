![StandardRL Components Logo](https://assets.standardrl.com/general/components/icon-full.png)
# MiniConv: Tiny Convolutional Networks for On-Device Environment Comprehension

MiniConv writes certain Convolutional Neural Networks (CNNs) as OpenGL ES 2 fragment shaders. This allows them to be evaluated on low-power, 'edge' devices. Using OpenGL ES 2 gives extremely wide compatibility. Performance on a Raspberry Pi 2W, a $10 device with around 1W of power usage and just 512MB of RAM, is given below. When used as a feature extractor in Reinforcement Learning over a network, large state space representations can be processed partly on edge devices, giving additional privacy properties, removing some processing requirements from servers and reducing the size of state transmissions. This approach, when deployed in a low bandwidth environment, provides improved decision latency over sending the full contents of large state spaces. Several RL environments exhibit no cost in performance by using CNNs that are compatible with MiniConv as a feature extractor.

# Model Training and Preparation

The Python file `embedder.py` includes the necessary functions to convert any PyTorch model which conforms to the following rules to be compiled as MiniConv model:

1. There are only ReLU, Conv2d and MaxPool2d layers

2. Every Conv2D or MaxPool2d layer must be followed by a ReLU

3. For both Conv2D and MaxPool2D there should be no specified padding

4. For both Conv2D and MaxPool2D the kernel size should be of equal width and height and should be 5 or less


## Example Usage

    import embedder
    import torch.nn as nn
    # Example usage:
    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(16, 4, kernel_size=2),
        nn.ReLU()
    )
    embedder.process_conv_layers(model, "xxx")

    

Once a model has been trained and prepared, you should have a directory with the following structure:

- /{`model_name`}
    - manifest.json
    -  {`model_name`}_{`layer_type`}NxN\_{`i`}\_{`n`}.glsl
    -   ...   

# Model Execution


The model executor is given as a C file, with the following dependencies:

- **cJSON.c**: JSON library for model manifest parsing.

- **libpng (-lpng)**: This is a library for handling PNG images.

- **zlib (-lz)**: This compression library is often required for PNG files as it is used to handle data compression.

- **math library (-lm)**: This is the standard math library in C.

- **EGL (-lEGL)**: Interface between rendering OpenGL and the underlying native platform windowing system.

- **OpenGL ES 2.0 (-lGLESv2)**: Support for OpenGL ES 2.0.

The following header files are also required for compilation to provide support for the JPEG input format:
1. **stb_image.h**: This is a header-only library for loading images in various formats (JPEG, PNG, BMP, TGA, etc.).

2. **stb_image_write.h**: This is a header-only library for saving images to disk in formats like PNG, BMP, TGA, and JPEG.

These stb libraries are self-contained and don’t require external linking with gcc (since they consist of single-header libraries), but you must ensure they are included in the project directory during compilation.

### To compile MiniConv:

`gcc cJSON.c loadmodel.c -o run -lpng -lz -lm -lEGL -lGLESv2`

### Usage
` ./run model_path input_path output_path input_width input_height`

# Evaluation and Performance
![Performance comparison between the same model on a Raspberry Pi 2W when running with MiniConv and when using the CPU using PyTorch.](https://assets.standardrl.com/general/components/miniconv/comparecpuandminiconv.png?)
*Performance comparison between the same model on a Raspberry Pi 2W when running with MiniConv and when using the CPU using PyTorch. The model has 3 layers and uses a kernel size of 3 on each. The improved performance of MiniConv allows for applications where live image feeds can be safely processed at a rate of 5 frames per second.*

![CPU Temperature and RAM usage on a Raspberry Pi 2W while running MiniConv.](https://assets.standardrl.com/general/components/miniconv/RAMandCPUTemp.png)
*System CPU Temperature and RAM usage on a Raspberry Pi 2W while running MiniConv. No external cooling was used. RAM usage remains relatively constant and low; 55% utilisation of the Pi Zero 2W's RAM is around 280MB. The RAM usage before and during execution is relatively similar, suggesting no significant memory pressure.*
