import os
import json
import torch
import torch.nn as nn

def check_model_rules(model):
    layers = list(model.children())  # Extract the list of layers
    valid = True
    error_messages = []
    
    # 1. Check for allowed layer types and their order
    for i, layer in enumerate(layers):
        # Rule 1: Only nn.ReLU, nn.Conv2d, and nn.MaxPool2d are allowed
        if not isinstance(layer, (nn.ReLU, nn.Conv2d, nn.MaxPool2d)):
            valid = False
            error_messages.append(f"Layer {i} is not a ReLU, Conv2d, or MaxPool2d.")
        
        # Rule 2: Conv2d or MaxPool2d must be followed by ReLU
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            # Ensure next layer exists and is a ReLU
            if i + 1 >= len(layers) or not isinstance(layers[i + 1], nn.ReLU):
                valid = False
                error_messages.append(f"Layer {i} ({layer.__class__.__name__}) is not followed by a ReLU.")
        
        # Rule 3 and 4: Check padding and kernel size for Conv2d and MaxPool2d
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            # Check padding is zero or None
            if layer.padding != (0, 0) and layer.padding != 0:
                valid = False
                error_messages.append(f"Layer {i} ({layer.__class__.__name__}) has non-zero padding. {layer.padding}")
            
            # Check kernel size is square and <= 5
            ks = layer.kernel_size
            kernel_size = ks if isinstance(ks, tuple) else (ks, ks)
            if kernel_size[0] != kernel_size[1] or kernel_size[0] > 5:
                valid = False
                error_messages.append(f"Layer {i} ({layer.__class__.__name__}) has a kernel size that is not square or is greater than 5.")

    # Return results
    if valid:
        print("Model follows all specified rules.")
    else:
        print("Model violates the following rules:")
        for msg in error_messages:
            print(msg)
            
    return valid
            
            
def get_max_pool_1():
    return """
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel))*inputTexel;

            highp float
                x0 = inputLoc.x,
                y0 = inputLoc.y;

            highp vec4 i0;

            gl_FragColor = fetch(images[0], x0, y0);
        }
    """

def get_max_pool_2():
    return """
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;

            highp float
                x0 = inputLoc.x - inputTexel.x,
                x1 = inputLoc.x,

                y0 = inputLoc.y - inputTexel.y,
                y1 = inputLoc.y;

            highp vec4 i0, i1, i2, i3;
            
            i0 = fetch(images[0], x0, y0);
            i1 = fetch(images[0], x1, y0);
            i2 = fetch(images[0], x0, y1);
            i3 = fetch(images[0], x1, y1);

            gl_FragColor = max(max(i0, i1), max(i2, i3));
        }
    """

def get_max_pool_3():
    return """
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;

            highp float
                x0 = inputLoc.x - inputTexel.x,
                x1 = inputLoc.x,
                x2 = inputLoc.x + inputTexel.x,

                y0 = inputLoc.y - inputTexel.y,
                y1 = inputLoc.y,
                y2 = inputLoc.y + inputTexel.y;

            highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8;

            i0 = fetch(images[0], x0, y0);
            i1 = fetch(images[0], x1, y0);
            i2 = fetch(images[0], x2, y0);
            i3 = fetch(images[0], x0, y1);
            i4 = fetch(images[0], x1, y1);
            i5 = fetch(images[0], x2, y1);
            i6 = fetch(images[0], x0, y2);
            i7 = fetch(images[0], x1, y2);
            i8 = fetch(images[0], x2, y2);

            gl_FragColor = max(max(max(max(i0, i1), max(i2, i3)), max(max(i4, i5), max(i6, i7))), i8);
        }
    """


def get_max_pool_4():
    return """
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(2.0, 2.0))*inputTexel;

        highp float
            x1 = inputLoc.x - inputTexel.x,
            x2 = inputLoc.x,
            x3 = inputLoc.x + inputTexel.x,
            x0 = inputLoc.x - inputTexel.x - inputTexel.x,

            y1 = inputLoc.y - inputTexel.y,
            y2 = inputLoc.y,
            y3 = inputLoc.y + inputTexel.y,
            y0 = inputLoc.y - inputTexel.y - inputTexel.y;

            highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;
            highp vec4 sum;
        
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x2, y0);
            i3 = fetch(images[%d], x3, y0);
            i4 = fetch(images[%d], x0, y1);
            i5 = fetch(images[%d], x1, y1);
            i6 = fetch(images[%d], x2, y1);
            i7 = fetch(images[%d], x3, y1);
            i8 = fetch(images[%d], x0, y2);
            i9 = fetch(images[%d], x1, y2);
            i10 = fetch(images[%d], x2, y2);
            i11 = fetch(images[%d], x3, y2);
            i12 = fetch(images[%d], x0, y3);
            i13 = fetch(images[%d], x1, y3);
            i14 = fetch(images[%d], x2, y3);
            i15 = fetch(images[%d], x3, y3);

            gl_FragColor = max(
                max(max(max(max(i0, i1), max(i2, i3)), max(max(i4, i5), max(i6, i7))), i8),
                max(max(max(max(i9, i10), max(i11, i12)), max(max(i13, i14), i15)))
            );
        }
    """


def get_max_pool_5():
    return """
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(2.0, 2.0))*inputTexel;

    
            highp float
                x1 = inputLoc.x - inputTexel.x,
                x2 = inputLoc.x,
                x3 = inputLoc.x + inputTexel.x,
                x0 = inputLoc.x - inputTexel.x - inputTexel.x,
                x4 = inputLoc.x + inputTexel.x + inputTexel.x,

                y1 = inputLoc.y - inputTexel.y,
                y2 = inputLoc.y,
                y3 = inputLoc.y + inputTexel.y,
                y0 = inputLoc.y - inputTexel.y - inputTexel.y,
                y4 = inputLoc.y + inputTexel.y + inputTexel.y;

            highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24;
            highp vec4 sum;
        
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x2, y0);
            i3 = fetch(images[%d], x3, y0);
            i4 = fetch(images[%d], x4, y0);
            i5 = fetch(images[%d], x0, y1);
            i6 = fetch(images[%d], x1, y1);
            i7 = fetch(images[%d], x2, y1);
            i8 = fetch(images[%d], x3, y1);
            i9 = fetch(images[%d], x4, y1);
            i10 = fetch(images[%d], x0, y2);
            i11 = fetch(images[%d], x1, y2);
            i12 = fetch(images[%d], x2, y2);
            i13 = fetch(images[%d], x3, y2);
            i14 = fetch(images[%d], x4, y2);
            i15 = fetch(images[%d], x0, y3);
            i16 = fetch(images[%d], x1, y3);
            i17 = fetch(images[%d], x2, y3);
            i18 = fetch(images[%d], x3, y3);
            i19 = fetch(images[%d], x4, y3);
            i20 = fetch(images[%d], x0, y4);
            i21 = fetch(images[%d], x1, y4);
            i22 = fetch(images[%d], x2, y4);
            i23 = fetch(images[%d], x3, y4);
            i24 = fetch(images[%d], x4, y4);

            gl_FragColor = max(max(
                max(max(max(max(i0, i1), max(i2, i3)), max(max(i4, i5), max(i6, i7))), i8),
                max(max(max(max(i9, i10), max(i11, i12)), max(max(i13, i14), i15)))
            ),
            max(
                max(max(max(i16, i17), max(max(i18, i19), max(i20, i21))), max(i22, max(i23, i24)))
            ));
        }
    """

def process_layer_1(layer, shaderNumber, fmt='%0.18f'):
    # some sanity check
    nInChannels = layer.weight.shape[1]
    outChannel = 4 * shaderNumber
    assert layer.weight.shape[2] == 1 and layer.weight.shape[3] == 1
    assert nInChannels % 4 == 0
    assert outChannel < layer.weight.shape[0]
    
    # start with the fragment shader main(..) code
    # declare input pixel coordinates (d1 is a uniform defined further on)
    code = """
        highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;
        
    
        highp float
            x0 = inputLoc.x,

            y0 = inputLoc.y;
            
        highp vec4 i0;
        highp vec4 sum;
    """
    
    # loop input textures        
    kernels = layer.weight.detach().numpy()[outChannel:outChannel+4,:,:,:]
    
    biases = layer.bias.detach().numpy()[outChannel:outChannel+4]
    numInputSamplers = nInChannels // 4
    
    for i in range(numInputSamplers):
        # sample input feature maps (function fetch(..) is defined later on)
        code += """
            i0 = fetch(images[%d], x0, y0);
        """ % tuple([i])

        # compute convolutions over these feature maps
        lanes = ["", "", "", ""]
        for shift in range(4):
            k = kernels[shift, 4*i:4*i+4,:,:].reshape(4)
            lanes[shift] += """
                + dot(vec4(%F, %F, %F, %F), i0)
            """.rstrip().replace('%F', fmt) % tuple(k)
        
        # sum them
        lanes = [_.strip()[2:] for _ in lanes]
        code += """
            sum %s vec4(
                %s,
                %s,
                %s,
                %s
            );
        """ % tuple(['=' if i == 0 else "+="] + lanes)        
    
    
    # write out the result
    code += """
        gl_FragColor = sum + vec4(%F, %F, %F, %F);
    """.replace('%F', fmt) % tuple(biases)
    
    # surround the convolution computing code with variable declarations,
    # brelu and luma sampling functions
    code = """
        uniform sampler2D images[%d];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            %s
        }
    """ % (numInputSamplers, code)
    
    return code

def process_layer_2(layer, shaderNumber, fmt='%0.18f'):
    # some sanity check
    nInChannels = layer.weight.shape[1]
    outChannel = 4 * shaderNumber
    assert layer.weight.shape[2] == 2 and layer.weight.shape[3] == 2
    assert nInChannels % 4 == 0
    assert outChannel < layer.weight.shape[0]
    
    # start with the fragment shader main(..) code
    # declare input pixel coordinates (d1 is a uniform defined further on)
    code = """
        highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;
        
    
        highp float
            x0 = inputLoc.x - inputTexel.x,
            x1 = inputLoc.x,

            y0 = inputLoc.y - inputTexel.y,
            y1 = inputLoc.y;
            
        highp vec4 i0, i1, i2, i3;
        highp vec4 sum;
    """
    
    # loop input textures        
    kernels = layer.weight.detach().numpy()[outChannel:outChannel+4,:,:,:]
    
    biases = layer.bias.detach().numpy()[outChannel:outChannel+4]
    numInputSamplers = nInChannels // 4
    
    for i in range(numInputSamplers):
        # sample input feature maps (function fetch(..) is defined later on)
        code += """
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x0, y1);
            i3 = fetch(images[%d], x1, y1);
        """ % tuple([i] * 4)

        # compute convolutions over these feature maps
        lanes = ["", "", "", ""]
        for shift in range(4):
            k = kernels[shift, 4*i:4*i+4,:,:].reshape(4*4)
            lanes[shift] += """
                + dot(vec4(%F, %F, %F, %F), i0)
                + dot(vec4(%F, %F, %F, %F), i1)
                + dot(vec4(%F, %F, %F, %F), i2)
                + dot(vec4(%F, %F, %F, %F), i3)
            """.rstrip().replace('%F', fmt) % tuple(k)
        
        # sum them
        lanes = [_.strip()[2:] for _ in lanes]
        code += """
            sum %s vec4(
                %s,
                %s,
                %s,
                %s
            );
        """ % tuple(['=' if i == 0 else "+="] + lanes)        
    
    
    # write out the result
    code += """
        gl_FragColor = sum + vec4(%F, %F, %F, %F);
    """.replace('%F', fmt) % tuple(biases)
    
    # surround the convolution computing code with variable declarations,
    # brelu and luma sampling functions
    code = """
        uniform sampler2D images[%d];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            %s
        }
    """ % (numInputSamplers, code)
    
    return code


def process_layer_3(layer, shaderNumber, fmt='%0.18f'):
    # some sanity check
    nInChannels = layer.weight.shape[1]
    outChannel = 4 * shaderNumber
    assert layer.weight.shape[2] == 3 and layer.weight.shape[3] == 3
    assert nInChannels % 4 == 0
    assert outChannel < layer.weight.shape[0]
    
    # start with the fragment shader main(..) code
    # declare input pixel coordinates (d1 is a uniform defined further on)
    code = """
        highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;
        
    
        highp float
            x0 = inputLoc.x - inputTexel.x,
            x1 = inputLoc.x,
            x2 = inputLoc.x + inputTexel.x,

            y0 = inputLoc.y - inputTexel.y,
            y1 = inputLoc.y,
            y2 = inputLoc.y + inputTexel.y;
            
        highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8;
        highp vec4 sum;
    """
    
    # loop input textures        
    kernels = layer.weight.detach().numpy()[outChannel:outChannel+4,:,:,:]
    
    biases = layer.bias.detach().numpy()[outChannel:outChannel+4]
    numInputSamplers = nInChannels // 4
    
    for i in range(numInputSamplers):
        # sample input feature maps (function fetch(..) is defined later on)
        code += """
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x2, y0);
            i3 = fetch(images[%d], x0, y1);
            i4 = fetch(images[%d], x1, y1);
            i5 = fetch(images[%d], x2, y1);
            i6 = fetch(images[%d], x0, y2);
            i7 = fetch(images[%d], x1, y2);
            i8 = fetch(images[%d], x2, y2);
        """ % tuple([i] * 9)

        # compute convolutions over these feature maps
        lanes = ["", "", "", ""]
        for shift in range(4):
            k = kernels[shift, 4*i:4*i+4,:,:].reshape(4*9)
            lanes[shift] += """
                + dot(vec4(%F, %F, %F, %F), i0)
                + dot(vec4(%F, %F, %F, %F), i1)
                + dot(vec4(%F, %F, %F, %F), i2)
                + dot(vec4(%F, %F, %F, %F), i3)
                + dot(vec4(%F, %F, %F, %F), i4)
                + dot(vec4(%F, %F, %F, %F), i5)
                + dot(vec4(%F, %F, %F, %F), i6)
                + dot(vec4(%F, %F, %F, %F), i7)
                + dot(vec4(%F, %F, %F, %F), i8)
            """.rstrip().replace('%F', fmt) % tuple(k)
        
        # sum them
        lanes = [_.strip()[2:] for _ in lanes]
        code += """
            sum %s vec4(
                %s,
                %s,
                %s,
                %s
            );
        """ % tuple(['=' if i == 0 else "+="] + lanes)        
    
    
    # write out the result
    code += """
        gl_FragColor = sum + vec4(%F, %F, %F, %F);
    """.replace('%F', fmt) % tuple(biases)
    
    # surround the convolution computing code with variable declarations,
    # brelu and luma sampling functions
    code = """
        uniform sampler2D images[%d];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            %s
        }
    """ % (numInputSamplers, code)
    
    return code



def process_layer_4(layer, shaderNumber, fmt='%0.18f'):
    # some sanity check
    nInChannels = layer.weight.shape[1]
    outChannel = 4 * shaderNumber
    assert layer.weight.shape[2] == 4 and layer.weight.shape[3] == 4
    assert nInChannels % 4 == 0
    assert outChannel < layer.weight.shape[0]
    
    # start with the fragment shader main(..) code
    # declare input pixel coordinates (d1 is a uniform defined further on)
    code = """
        highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(2.0, 2.0))*inputTexel;
        
    
        highp float
            x1 = inputLoc.x - inputTexel.x,
            x2 = inputLoc.x,
            x3 = inputLoc.x + inputTexel.x,
            x0 = inputLoc.x - inputTexel.x - inputTexel.x,

            y1 = inputLoc.y - inputTexel.y,
            y2 = inputLoc.y,
            y3 = inputLoc.y + inputTexel.y,
            y0 = inputLoc.y - inputTexel.y - inputTexel.y;
            
        highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;
        highp vec4 sum;
    """
    
    # loop input textures        
    kernels = layer.weight.detach().numpy()[outChannel:outChannel+4,:,:,:]
    
    biases = layer.bias.detach().numpy()[outChannel:outChannel+4]
    numInputSamplers = nInChannels // 4
    
    for i in range(numInputSamplers):
        # sample input feature maps (function fetch(..) is defined later on)
        code += """
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x2, y0);
            i3 = fetch(images[%d], x3, y0);
            i4 = fetch(images[%d], x0, y1);
            i5 = fetch(images[%d], x1, y1);
            i6 = fetch(images[%d], x2, y1);
            i7 = fetch(images[%d], x3, y1);
            i8 = fetch(images[%d], x0, y2);
            i9 = fetch(images[%d], x1, y2);
            i10 = fetch(images[%d], x2, y2);
            i11 = fetch(images[%d], x3, y2);
            i12 = fetch(images[%d], x0, y3);
            i13 = fetch(images[%d], x1, y3);
            i14 = fetch(images[%d], x2, y3);
            i15 = fetch(images[%d], x3, y3);
        """ % tuple([i] * 16)

        # compute convolutions over these feature maps
        lanes = ["", "", "", ""]
        for shift in range(4):
            k = kernels[shift, 4*i:4*i+4,:,:].reshape(4*16)
            lanes[shift] += """
                + dot(vec4(%F, %F, %F, %F), i0)
                + dot(vec4(%F, %F, %F, %F), i1)
                + dot(vec4(%F, %F, %F, %F), i2)
                + dot(vec4(%F, %F, %F, %F), i3)
                + dot(vec4(%F, %F, %F, %F), i4)
                + dot(vec4(%F, %F, %F, %F), i5)
                + dot(vec4(%F, %F, %F, %F), i6)
                + dot(vec4(%F, %F, %F, %F), i7)
                + dot(vec4(%F, %F, %F, %F), i8)
                + dot(vec4(%F, %F, %F, %F), i9)
                + dot(vec4(%F, %F, %F, %F), i10)
                + dot(vec4(%F, %F, %F, %F), i11)
                + dot(vec4(%F, %F, %F, %F), i12)
                + dot(vec4(%F, %F, %F, %F), i13)
                + dot(vec4(%F, %F, %F, %F), i14)
                + dot(vec4(%F, %F, %F, %F), i15)
            """.rstrip().replace('%F', fmt) % tuple(k)
        
        # sum them
        lanes = [_.strip()[2:] for _ in lanes]
        code += """
            sum %s vec4(
                %s,
                %s,
                %s,
                %s
            );
        """ % tuple(['=' if i == 0 else "+="] + lanes)        
    
    
    # write out the result
    code += """
        gl_FragColor = sum + vec4(%F, %F, %F, %F);
    """.replace('%F', fmt) % tuple(biases)
    
    # surround the convolution computing code with variable declarations,
    # brelu and luma sampling functions
    code = """
        uniform sampler2D images[%d];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            %s
        }
    """ % (numInputSamplers, code)
    
    return code




def process_layer_5(layer, shaderNumber, fmt='%0.18f'):
    # some sanity check
    nInChannels = layer.weight.shape[1]
    outChannel = 4 * shaderNumber
    assert layer.weight.shape[2] == 5 and layer.weight.shape[3] == 5
    assert nInChannels % 4 == 0
    assert outChannel < layer.weight.shape[0]
    
    # start with the fragment shader main(..) code
    # declare input pixel coordinates (d1 is a uniform defined further on)
    code = """
        highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(2.0, 2.0))*inputTexel;
        
    
        highp float
            x1 = inputLoc.x - inputTexel.x,
            x2 = inputLoc.x,
            x3 = inputLoc.x + inputTexel.x,
            x0 = inputLoc.x - inputTexel.x - inputTexel.x,
            x4 = inputLoc.x + inputTexel.x + inputTexel.x,

            y1 = inputLoc.y - inputTexel.y,
            y2 = inputLoc.y,
            y3 = inputLoc.y + inputTexel.y,
            y0 = inputLoc.y - inputTexel.y - inputTexel.y,
            y4 = inputLoc.y + inputTexel.y + inputTexel.y;
            
        highp vec4 i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24;
        highp vec4 sum;
    """
    
    # loop input textures        
    kernels = layer.weight.detach().numpy()[outChannel:outChannel+4,:,:,:]
    
    biases = layer.bias.detach().numpy()[outChannel:outChannel+4]
    numInputSamplers = nInChannels // 4
    
    for i in range(numInputSamplers):
        # sample input feature maps (function fetch(..) is defined later on)
        code += """
            i0 = fetch(images[%d], x0, y0);
            i1 = fetch(images[%d], x1, y0);
            i2 = fetch(images[%d], x2, y0);
            i3 = fetch(images[%d], x3, y0);
            i4 = fetch(images[%d], x4, y0);
            i5 = fetch(images[%d], x0, y1);
            i6 = fetch(images[%d], x1, y1);
            i7 = fetch(images[%d], x2, y1);
            i8 = fetch(images[%d], x3, y1);
            i9 = fetch(images[%d], x4, y1);
            i10 = fetch(images[%d], x0, y2);
            i11 = fetch(images[%d], x1, y2);
            i12 = fetch(images[%d], x2, y2);
            i13 = fetch(images[%d], x3, y2);
            i14 = fetch(images[%d], x4, y2);
            i15 = fetch(images[%d], x0, y3);
            i16 = fetch(images[%d], x1, y3);
            i17 = fetch(images[%d], x2, y3);
            i18 = fetch(images[%d], x3, y3);
            i19 = fetch(images[%d], x4, y3);
            i20 = fetch(images[%d], x0, y4);
            i21 = fetch(images[%d], x1, y4);
            i22 = fetch(images[%d], x2, y4);
            i23 = fetch(images[%d], x3, y4);
            i24 = fetch(images[%d], x4, y4);
        """ % tuple([i] * 25)

        # compute convolutions over these feature maps
        lanes = ["", "", "", ""]
        for shift in range(4):
            k = kernels[shift, 4*i:4*i+4,:,:].reshape(4*25)
            lanes[shift] += """
                + dot(vec4(%F, %F, %F, %F), i0)
                + dot(vec4(%F, %F, %F, %F), i1)
                + dot(vec4(%F, %F, %F, %F), i2)
                + dot(vec4(%F, %F, %F, %F), i3)
                + dot(vec4(%F, %F, %F, %F), i4)
                + dot(vec4(%F, %F, %F, %F), i5)
                + dot(vec4(%F, %F, %F, %F), i6)
                + dot(vec4(%F, %F, %F, %F), i7)
                + dot(vec4(%F, %F, %F, %F), i8)
                + dot(vec4(%F, %F, %F, %F), i9)
                + dot(vec4(%F, %F, %F, %F), i10)
                + dot(vec4(%F, %F, %F, %F), i11)
                + dot(vec4(%F, %F, %F, %F), i12)
                + dot(vec4(%F, %F, %F, %F), i13)
                + dot(vec4(%F, %F, %F, %F), i14)
                + dot(vec4(%F, %F, %F, %F), i15)
                + dot(vec4(%F, %F, %F, %F), i16)
                + dot(vec4(%F, %F, %F, %F), i17)
                + dot(vec4(%F, %F, %F, %F), i18)
                + dot(vec4(%F, %F, %F, %F), i19)
                + dot(vec4(%F, %F, %F, %F), i20)
                + dot(vec4(%F, %F, %F, %F), i21)
                + dot(vec4(%F, %F, %F, %F), i22)
                + dot(vec4(%F, %F, %F, %F), i23)
                + dot(vec4(%F, %F, %F, %F), i24)
            """.rstrip().replace('%F', fmt) % tuple(k)
        
        # sum them
        lanes = [_.strip()[2:] for _ in lanes]
        code += """
            sum %s vec4(
                %s,
                %s,
                %s,
                %s
            );
        """ % tuple(['=' if i == 0 else "+="] + lanes)        
    
    
    # write out the result
    code += """
        gl_FragColor = sum + vec4(%F, %F, %F, %F);
    """.replace('%F', fmt) % tuple(biases)
    
    # surround the convolution computing code with variable declarations,
    # brelu and luma sampling functions
    code = """
        uniform sampler2D images[%d];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            %s
        }
    """ % (numInputSamplers, code)
    
    return code




# Dictionary to dynamically select the correct function based on kernel size
process_layer_functions = {
    1: process_layer_1,
    2: process_layer_2,
    3: process_layer_3,
    4: process_layer_4,
    5: process_layer_5
}

max_pool_functions = {
    1: get_max_pool_1,
    2: get_max_pool_2,
    3: get_max_pool_3,
    4: get_max_pool_4,
    5: get_max_pool_5
}

def process_conv_layers(model, model_name="newmodel"):
    if not check_model_rules(model):
        raise Exception("Model not compatiable")
        
    # Create directory for model if it does not exist
    os.makedirs(model_name, exist_ok=True)

    manifest = []
    last_conv_out_channels = None  # Variable to track output channels of last Conv2d layer

    for idx, layer in enumerate(model):
        entry = {}

        # Check if the layer is a Conv2d layer
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]  # Assuming kernel size is square

            # Ensure kernel size is within the allowed range
            if kernel_size in process_layer_functions:
                # Get the processing function for the current kernel size
                process_layer = process_layer_functions[kernel_size]

                # Determine the number of output channel sets
                out_channels = layer.out_channels
                in_channels = layer.in_channels
                # --- Pi 4 (VideoCore VI) resource check ------------------------
                textures  = in_channels // 4
                samples   = (kernel_size ** 2) * textures
                if textures > 8 or samples > 64:
                    raise ValueError(
                        f"Layer {idx}: Cin={in_channels}, k={kernel_size} "
                        f"needs {textures} textures / {samples} samples > GPU limit"
                    )
                # ----------------------------------------------------------------

                last_conv_out_channels = out_channels  # Update last conv output channels
                if out_channels % 4 != 0:
                    raise ValueError("Number of output channels must be a multiple of 4")

                num_sets = out_channels // 4
                shader_paths = []

                # Call the processing function for each set of 4 output channels
                for i in range(num_sets):
                    output_text = process_layer(layer, i)
                    # Define filename according to the specification
                    filename = f"{model_name}/{model_name}_Conv{kernel_size}x{kernel_size}_{idx}_{i}.glsl"
                    shader_paths.append(f"{model_name}_Conv{kernel_size}x{kernel_size}_{idx}_{i}.glsl")
                    with open(filename, "w") as file:
                        file.write(output_text)

                # Add entry to manifest
                entry = {
                    "name": f"layer {idx} (Conv{kernel_size}x{kernel_size})",
                    "type": "conv",
                    "shader": shader_paths,
                    "stride": layer.stride[0],  # assuming square stride
                    "kernel": kernel_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels
                }

        # Check if the layer is a MaxPool layer
        elif isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            if k in max_pool_functions:
                # Guard: kernel size and stride *must* match (GLSL expects it)
                assert layer.stride == (k, k) or layer.stride == k, \
                    f"MaxPool at layer {idx} has stride {layer.stride}, kernel {k}"

                get_max_pool = max_pool_functions[k]
                output_text = get_max_pool()

                filename    = f"{model_name}/{model_name}_MaxPool{k}x{k}_{idx}.glsl"
                shader_path = f"{model_name}_MaxPool{k}x{k}_{idx}.glsl"
                with open(filename, "w") as file:
                    file.write(output_text)
                    
                num_sets = last_conv_out_channels // 4

                # Add entry to manifest, using last Conv2d out_channels for both in and out channels
                entry = {
                    "name": f"layer {idx} (MaxPool{kernel_size}x{kernel_size})",
                    "type": "maxpool",
                    "shader": [shader_path]*num_sets,
                    "stride": layer.stride,  # MaxPool stride is typically square
                    "kernel": kernel_size,
                    "in_channels": last_conv_out_channels,
                    "out_channels": last_conv_out_channels
                }

        # Append entry if it has been populated
        if entry:
            manifest.append(entry)

    # Save the manifest to manifest.json
    manifest_path = os.path.join(model_name, "manifest.json")
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=4)
