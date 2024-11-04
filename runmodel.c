#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <unistd.h>

// Define constants for maximum sizes
#define MAX_LAYERS 100
#define MAX_SHADERS_PER_LAYER 10
#define MAX_SHADER_PATH_LENGTH 256

// Define the layer struct with a list of shader paths
typedef struct {
    char name[256];
    char *type;       // Layer type (e.g., "maxpool", "conv")
    int stride;
    int kernel;
    int in_channels;
    int out_channels;
    char shaders[MAX_SHADERS_PER_LAYER][MAX_SHADER_PATH_LENGTH];  // Array of shader paths
    int num_shaders;  // Number of shaders in this layer
} layer;


// Define the programbank struct to hold compiled programs for each layer
typedef struct {
    GLuint programs[MAX_SHADERS_PER_LAYER];  // Array of program IDs
    int num_programs;  // Number of programs in this layer
} programbank;

// Define the texturebank struct
typedef struct {
    int num_textures;
    GLuint textures[100];  // Adjust as needed
} texturebank;




int load_layers_from_json(const char *directory, layer layers[], int max_layers) {
    // Construct the path to manifest.json
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", directory);

    FILE *file = fopen(manifest_path, "r");
    if (!file) {
        perror("Unable to open manifest.json");
        return -1;
    }

    // Read file contents into a buffer
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = (char *)malloc(length + 1);
    if (!data) {
        fclose(file);
        perror("Memory allocation failed");
        return -1;
    }
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    // Parse JSON
    cJSON *json = cJSON_Parse(data);
    free(data);
    if (!json) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        return -1;
    }

    int layer_count = 0;
    cJSON *layer_json = NULL;
    cJSON_ArrayForEach(layer_json, json) {
        if (layer_count >= max_layers) {
            fprintf(stderr, "Max layers exceeded\n");
            break;
        }

        // Parse each field of the layer
        cJSON *name = cJSON_GetObjectItem(layer_json, "name");
        cJSON *stride = cJSON_GetObjectItem(layer_json, "stride");
        cJSON *kernel = cJSON_GetObjectItem(layer_json, "kernel");
        cJSON *in_channels = cJSON_GetObjectItem(layer_json, "in_channels");
        cJSON *out_channels = cJSON_GetObjectItem(layer_json, "out_channels");
        cJSON *shaders = cJSON_GetObjectItem(layer_json, "shader");
        cJSON *type_json = cJSON_GetObjectItemCaseSensitive(layer_json, "type");

        // Populate the layer struct
        if (name) strncpy(layers[layer_count].name, name->valuestring, sizeof(layers[layer_count].name) - 1);
        if (stride) layers[layer_count].stride = stride->valueint;
        if (kernel) layers[layer_count].kernel = kernel->valueint;
        if (in_channels) layers[layer_count].in_channels = in_channels->valueint;
        if (out_channels) layers[layer_count].out_channels = out_channels->valueint;

        // Populate the type field
        if (cJSON_IsString(type_json) && (type_json->valuestring != NULL)) {
            layers[layer_count].type = strdup(type_json->valuestring);
        } else {
            layers[layer_count].type = strdup("conv"); // Default to "conv" if type is missing
        }

        // Populate shader paths from JSON list
        layers[layer_count].num_shaders = 0;
        cJSON *shader_path;
        cJSON_ArrayForEach(shader_path, shaders) {
            if (layers[layer_count].num_shaders < MAX_SHADERS_PER_LAYER) {
                // Construct the full shader path
                char full_shader_path[MAX_SHADER_PATH_LENGTH];
                snprintf(full_shader_path, sizeof(full_shader_path), "%s/%s", directory, shader_path->valuestring);
                strncpy(layers[layer_count].shaders[layers[layer_count].num_shaders], full_shader_path, MAX_SHADER_PATH_LENGTH - 1);
                layers[layer_count].num_shaders++;
            }
        }

        layer_count++;
    }

    cJSON_Delete(json);
    return layer_count;
}


// Function to read the shader file
char* ReadShaderFile(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open shader file: %s\n", filePath);
        return NULL;
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the shader source
    char* shaderSource = (char*)malloc(length + 1);
    if (!shaderSource) {
        fprintf(stderr, "Error: Could not allocate memory for shader source\n");
        fclose(file);
        return NULL;
    }

    // Read the file content into shaderSource
    fread(shaderSource, 1, length, file);
    shaderSource[length] = '\0'; // Null-terminate the string

    fclose(file);
    return shaderSource;
}

// Function to compile a shader
GLuint CompileShader(GLenum shaderType, const char* shaderSource) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        char* log = (char*)malloc(logLength);
        glGetShaderInfoLog(shader, logLength, NULL, log);
        fprintf(stderr, "Error: Shader compilation failed: %s\n", log);
        free(log);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

// Function to load and compile shaders
GLuint LoadShader(const char* vertexFilePath, const char* fragmentFilePath) {
    // Read shader files
    char* vertexShaderSource = ReadShaderFile(vertexFilePath);
    char* fragmentShaderSource = ReadShaderFile(fragmentFilePath);

    if (!vertexShaderSource || !fragmentShaderSource) {
        free(vertexShaderSource);
        free(fragmentShaderSource);
        return 0; // Return 0 if reading failed
    }

    // Compile shaders
    GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Create a shader program and link the shaders
    GLuint program = glCreateProgram();
    if (vertexShader) glAttachShader(program, vertexShader);
    if (fragmentShader) glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check for linking errors
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        char* log = (char*)malloc(logLength);
        glGetProgramInfoLog(program, logLength, NULL, log);
        fprintf(stderr, "Error: Program linking failed: %s\n", log);
        free(log);
        glDeleteProgram(program);
        program = 0; // Set program to 0 on failure
    }

    // Cleanup shaders as they are linked into the program now
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    free(vertexShaderSource);
    free(fragmentShaderSource);

    return program;
}



// Function to print all layers
void print_layers(const layer layers[], int count) {
    for (int i = 0; i < count; i++) {
        printf("Layer %d:\n", i + 1);
        printf("  Name: %s\n", layers[i].name);
        printf("  Type: %s\n", layers[i].type);
        printf("  Stride: %d\n", layers[i].stride);
        printf("  Kernel: %d\n", layers[i].kernel);
        printf("  In Channels: %d\n", layers[i].in_channels);
        printf("  Out Channels: %d\n", layers[i].out_channels);

        // Print the shaders for this layer
        printf("  Shaders:\n");
        for (int j = 0; j < layers[i].num_shaders; j++) {
            printf("    %s\n", layers[i].shaders[j]);
        }
        
        printf("\n");
    }
}
void initialize_programbanks(layer layers[], programbank programbanks[], int layer_count) {
    for (int i = 0; i < layer_count; i++) {
        programbanks[i].num_programs = layers[i].num_shaders;
        for (int j = 0; j < layers[i].num_shaders; j++) {
            programbanks[i].programs[j] = LoadShader("../vertex_shader.glsl", layers[i].shaders[j]);
        }
    }
}




void prep_layers(layer layers[], programbank programbanks[], texturebank texturebanks[], int dimensions[][3], int count) {

    for (int i = 0; i < count; i++) {
        programbank *pb = &programbanks[i];
        layer *current_layer = &layers[i];
        
        // Get input width, height, and channels from dimensions array for the current layer
        int input_width = dimensions[i][0];
        int input_height = dimensions[i][1];

        // Calculate output width and height based on stride and kernel size
        int output_width = (input_width - current_layer->kernel) / current_layer->stride + 1;
        int output_height = (input_height - current_layer->kernel) / current_layer->stride + 1;

        // Compute inputTexel and outputTexel for shader
        float inputTexel[2] = { current_layer->stride * 1.0f / input_width, current_layer->stride * 1.0f / input_height };
        float outputTexel[2] = { 1.0f / output_width, 1.0f / output_height };

        int num_programs = pb->num_programs;
        int num_output_textures = texturebanks[i + 1].num_textures;

        //printf("This layer has %d output textures and %d programs \n", num_output_textures, num_programs);

        for (int p = 0; p < num_programs; p++) {
            glUseProgram(pb->programs[p]);

            // Get and set the 'inputTexel' uniform
            GLint inputTexel_location = glGetUniformLocation(pb->programs[p], "inputTexel");
            if (inputTexel_location != -1) {
                glUniform2fv(inputTexel_location, 1, inputTexel);
            }


            // Get and set the 'outputTexel' uniform
            GLint outputTexel_location = glGetUniformLocation(pb->programs[p], "outputTexel");
            if (outputTexel_location != -1) {
                glUniform2fv(outputTexel_location, 1, outputTexel);
            }

            // Bind textures as before, render as before
            int num_textures = texturebanks[i].num_textures;
            GLint uniform_location = glGetUniformLocation(pb->programs[p], "images");

            if (uniform_location == -1) {
                fprintf(stderr, "Could not find uniform 'images' in shader program %d\n", pb->programs[p]);
                //continue;
            }
            if (strcmp(current_layer->type, "maxpool") == 0) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texturebanks[i].textures[p]);
                glUniform1i(uniform_location, 0);
            }else{
                // Bind each texture in texturebanks[i] and assign them to the sampler array
                for (int j = 0; j < num_textures; j++) {
                    glActiveTexture(GL_TEXTURE0 + j);
                    glBindTexture(GL_TEXTURE_2D, texturebanks[i].textures[j]);
                    glUniform1i(uniform_location + j, j);
                }
            }

        }
    }

    glUseProgram(0);
}





void render_layers(layer layers[], programbank programbanks[], texturebank texturebanks[], int dimensions[][3], int count) {

    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f
    };
    GLuint indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    // Generate and bind VBO
    GLuint VBO, EBO;
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // Bind and set VBO (for vertex data)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // Bind and set EBO (for indices)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);


    for (int i = 0; i < count; i++) {
        programbank *pb = &programbanks[i];
        layer *current_layer = &layers[i];
        
        // Get input width, height, and channels from dimensions array for the current layer
        int input_width = dimensions[i][0];
        int input_height = dimensions[i][1];

        // Calculate output width and height based on stride and kernel size
        int output_width = (input_width - current_layer->kernel) / current_layer->stride + 1;
        int output_height = (input_height - current_layer->kernel) / current_layer->stride + 1;

        // Compute inputTexel and outputTexel for shader
        float inputTexel[2] = { current_layer->stride * 1.0f / input_width, current_layer->stride * 1.0f / input_height };
        float outputTexel[2] = { 1.0f / output_width, 1.0f / output_height };

        int num_programs = pb->num_programs;
        int num_output_textures = texturebanks[i + 1].num_textures;

        //printf("This layer has %d output textures and %d programs \n", num_output_textures, num_programs);

        for (int p = 0; p < num_programs; p++) {
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(pb->programs[p]);

            /*
            // Get and set the 'inputTexel' uniform
            GLint inputTexel_location = glGetUniformLocation(pb->programs[p], "inputTexel");
            if (inputTexel_location != -1) {
                glUniform2fv(inputTexel_location, 1, inputTexel);
            }


            // Get and set the 'outputTexel' uniform
            GLint outputTexel_location = glGetUniformLocation(pb->programs[p], "outputTexel");
            if (outputTexel_location != -1) {
                glUniform2fv(outputTexel_location, 1, outputTexel);
            }
            */
            // Bind textures as before, render as before
            int num_textures = texturebanks[i].num_textures;
            GLint uniform_location = glGetUniformLocation(pb->programs[p], "images");

            if (uniform_location == -1) {
                fprintf(stderr, "Could not find uniform 'images' in shader program %d\n", pb->programs[p]);
                //continue;
            }


            if (strcmp(current_layer->type, "maxpool") == 0) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texturebanks[i].textures[p]);
                glUniform1i(uniform_location, 0);
            }else{
                // Bind each texture in texturebanks[i] and assign them to the sampler array
                for (int j = 0; j < num_textures; j++) {
                    glActiveTexture(GL_TEXTURE0 + j);
                    glBindTexture(GL_TEXTURE_2D, texturebanks[i].textures[j]);
                    glUniform1i(uniform_location + j, j);
                }
            }
            

            // Render to each output texture in texturebanks[i+1]
            int output_texture = texturebanks[i + 1].textures[p];

            GLuint fbo;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output_texture, 0);

            //printf("Binding texture %d to framebuffer \n", output_texture);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                fprintf(stderr, "Framebuffer is not complete for layer %d texture %d\n", i, p);
                continue;
            }
            //printf("Output size is %d by %d \n", output_width, output_height);

            glViewport(0, 0, output_width, output_height);  // Set the viewport to the output size

            // Bind and draw the quad as before (assuming you've set up VBOs for vertices and texCoords)
            // glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            // Define vertex attribute (position)
            GLuint positionAttributeVertexPos = glGetAttribLocation(pb->programs[p], "aPos");
            glEnableVertexAttribArray(positionAttributeVertexPos);
            glVertexAttribPointer(positionAttributeVertexPos, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);

            // Set the vertex attributes: texture coordinates
            GLuint positionAttributeTextureCoor = glGetAttribLocation(pb->programs[p], "aTexCoord");
            glEnableVertexAttribArray(positionAttributeTextureCoor);
            glVertexAttribPointer(positionAttributeTextureCoor, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(1, &fbo);
        }
    }

    glUseProgram(0);
}



// Function to initialize EGL and OpenGL ES 2.0
EGLDisplay initializeEGL(EGLDisplay* eglDisplay, EGLContext* eglContext, EGLSurface* eglSurface) {
    // Get an EGL display connection
    *eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (*eglDisplay == EGL_NO_DISPLAY) {
        fprintf(stderr, "Error: Unable to open connection to local windowing system.\n");
        return EGL_NO_DISPLAY;
    }
    // Initialize the EGL display connection
    if (!eglInitialize(*eglDisplay, NULL, NULL)) {
        fprintf(stderr, "Error: Unable to initialize EGL.\n");
        return EGL_NO_DISPLAY;
    }
    // Choose a suitable EGL framebuffer configuration
    EGLint attribList[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_NONE
    };
    EGLConfig eglConfig;
    EGLint numConfigs;
    if (!eglChooseConfig(*eglDisplay, attribList, &eglConfig, 1, &numConfigs)) {
        fprintf(stderr, "Error: Unable to choose EGL config.\n");
        return EGL_NO_DISPLAY;
    }
    // Create an EGL window surface (Replace with eglCreatePbufferSurface if using offscreen)
    *eglSurface = eglCreatePbufferSurface(*eglDisplay, eglConfig, NULL);
    if (*eglSurface == EGL_NO_SURFACE) {
        fprintf(stderr, "Error: Unable to create EGL window surface.\n");
        return EGL_NO_DISPLAY;
    }
    // Create an OpenGL ES 2.0 context
    EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    *eglContext = eglCreateContext(*eglDisplay, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if (*eglContext == EGL_NO_CONTEXT) {
        fprintf(stderr, "Error: Unable to create OpenGL ES context.\n");
        return EGL_NO_DISPLAY;
    }
    // Make the EGL context and surface current
    if (!eglMakeCurrent(*eglDisplay, *eglSurface, *eglSurface, *eglContext)) {
        fprintf(stderr, "Error: Unable to make EGL context and surface current.\n");
        return EGL_NO_DISPLAY;
    }
    printf("EGL and OpenGL ES 2.0 initialized successfully.\n");
    return *eglDisplay;
}

void save_texture_to_csv(unsigned char *data, int width, int height) {
    // Allocate arrays for each channel
    float *redChannel = (float *)malloc(width * height * sizeof(float));
    float *greenChannel = (float *)malloc(width * height * sizeof(float));
    float *blueChannel = (float *)malloc(width * height * sizeof(float));
    float *alphaChannel = (float *)malloc(width * height * sizeof(float));

    if (!redChannel || !greenChannel || !blueChannel || !alphaChannel) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(redChannel);
        free(greenChannel);
        free(blueChannel);
        free(alphaChannel);
        return;
    }

    // Convert unsigned char data to float values in the range [0, 1]
    for (int i = 0; i < width * height; i++) {
        redChannel[i] = data[i * 4] / 255.0f;     // R
        greenChannel[i] = data[i * 4 + 1] / 255.0f; // G
        blueChannel[i] = data[i * 4 + 2] / 255.0f;  // B
        alphaChannel[i] = data[i * 4 + 3] / 255.0f; // A
    }

    // Write each channel to its respective CSV file
    const char *filenames[4] = {"red_channel.csv", "green_channel.csv", "blue_channel.csv", "alpha_channel.csv"};
    float *channels[4] = {redChannel, greenChannel, blueChannel, alphaChannel};

    for (int c = 0; c < 4; c++) {
        FILE *file = fopen(filenames[c], "w");
        if (!file) {
            fprintf(stderr, "Error opening file %s for writing.\n", filenames[c]);
            continue;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Write float values to CSV, formatted to 6 decimal places
                fprintf(file, "%.6f", channels[c][y * width + x]);
                if (x < width - 1) {
                    fprintf(file, ","); // Separate with commas
                }
            }
            fprintf(file, "\n"); // New line after each row
        }
        fclose(file);
    }

    // Free allocated memory
    free(redChannel);
    free(greenChannel);
    free(blueChannel);
    free(alphaChannel);
}


void save_final_texture_as_jpeg(texturebank *last_texturebank, int output_width, int output_height) {
    // Check if there’s exactly one texture in the last texturebank
    if (last_texturebank->num_textures != 1) {
        printf("The final layer has multiple textures. Cannot save as a single image.\n");
        return;
    }

    GLuint final_texture = last_texturebank->textures[0];


    printf("Final texture is %d \n", final_texture);

    // Create a framebuffer and attach the final texture
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, final_texture, 0);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer incomplete. Cannot read texture.\n");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        return;
    }

    // Allocate memory to read pixels
    unsigned char *data = (unsigned char *)malloc(4 * output_width * output_height); // RGBA format
    if (!data) {
        fprintf(stderr, "Memory allocation failed.\n");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        return;
    }

    // Read pixels from the framebuffer
    glReadPixels(0, 0, output_width, output_height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    save_texture_to_csv(data, output_width, output_height);

    // Save the image as JPEG (stb_image_write flips images vertically by default)
    if (stbi_write_jpg("output.jpg", output_width, output_height, 4, data, 90)) {
        printf("Image saved as output.jpg\n");
    } else {
        fprintf(stderr, "Failed to save image.\n");
    }

    // Clean up
    free(data);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
}

// Cleanup function to free allocated memory
void cleanup_resources(layer *layers, int num_layers, programbank *programbanks, int num_programbanks, texturebank *texturebanks, int num_texturebanks) {
    // Cleanup layers
    printf("Freeing all resources\n");


    printf("Freed layers\n");

    // Cleanup programbanks
    for (int i = 0; i < num_programbanks; i++) {
        for (int j = 0; j < programbanks[i].num_programs; j++){
            glDeleteProgram(programbanks[i].programs[j]); // Delete OpenGL programs
        }
    }
    printf("Freed programbanks\n");

    // Cleanup texturebanks
    for (int i = 0; i < num_texturebanks; i++) {
        glDeleteTextures(texturebanks[i].num_textures, texturebanks[i].textures); // Delete OpenGL textures
    }
    printf("Freed texturebanks\n");
}


int main(int argc, char *argv[]) {

    if (argc != 6) {
        fprintf(stderr, "Usage: %s <path_to_model> <path_to_input_file> <path_to_output> <input_width> <input_height>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *modelpath = argv[1];
    const char *inputpath = argv[2];
    const char *outputfile = argv[3];

    int width = atoi(argv[4]);  // Convert the count argument to an integer
    int height = atoi(argv[5]);  // Convert the count argument to an integer

    EGLDisplay eglDisplay;
    EGLContext eglContext;
    EGLSurface eglSurface;

    // Initialize EGL and OpenGL ES
    if (initializeEGL(&eglDisplay, &eglContext, &eglSurface) == EGL_NO_DISPLAY) {
        return -1;
    }

    layer layers[MAX_LAYERS];
    int count = load_layers_from_json(modelpath, layers, MAX_LAYERS);
    if (count <= 0) {
        printf("No layers loaded.\n");
        return 1;
    }

    print_layers(layers, count);

    programbank programbanks[MAX_LAYERS];
    initialize_programbanks(layers, programbanks, count);

    // Load image.jpg as the initial input texture
    int input_width, input_height, input_channels;

    /* Optional: Load from JPG
        unsigned char *image_data = stbi_load(inputpath, &input_width, &input_height, &input_channels, 4);  // Force RGBA (4 channels)
        if (!image_data) {
            fprintf(stderr, "Failed to load %s\n", inputpath);
            return 1;
        }
    */
    
    FILE *fileptr = fopen(inputpath, "rb");  // Open the file in binary mode

    unsigned char* image_data = (unsigned char *)malloc(width*height*4 * sizeof(unsigned char)); // Enough memory for the file
    fread(image_data, width*height*4 * sizeof(unsigned char), 1, fileptr); // Read in the entire file
    fclose(fileptr); // Close the file

    if (!image_data) {
        fprintf(stderr, "Failed to load %s\n", inputpath);
        return 1;
    }

    // Dimensions array to store width, height, and channels
    int dimensions[3 * (count + 1)];
    dimensions[0] = input_width;
    dimensions[1] = input_height;
    dimensions[2] = input_channels;

    int current_width = input_width;
    int current_height = input_height;

    // Texturebanks array with one extra element
    texturebank texturebanks[MAX_LAYERS + 1];

    // Create the first texturebank for input image
    texturebanks[0].num_textures = 1;
    glGenTextures(1, texturebanks[0].textures);
    glBindTexture(GL_TEXTURE_2D, texturebanks[0].textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, input_width, input_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    /* Free image data after loading to texture
        stbi_image_free(image_data);
    */
   free(image_data);

    // Iterate through each layer and calculate new dimensions
    for (int i = 0; i < count; i++) {
        int stride = layers[i].stride;
        int kernel = layers[i].kernel;

        int new_width = (current_width - kernel) / stride + 1;
        int new_height = (current_height - kernel) / stride + 1;
        int new_channels = layers[i].out_channels;

        // Store intermediate dimensions in the array
        dimensions[3 * (i + 1)] = new_width;
        dimensions[3 * (i + 1) + 1] = new_height;
        dimensions[3 * (i + 1) + 2] = new_channels;

        // Calculate number of textures needed
        int num_textures = (new_channels + 3) / 4;  // One texture per 4 channels

        // Create the texturebank for this layer
        texturebanks[i + 1].num_textures = num_textures;
        glGenTextures(num_textures, texturebanks[i + 1].textures);

        for (int j = 0; j < num_textures; j++) {
            glBindTexture(GL_TEXTURE_2D, texturebanks[i + 1].textures[j]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, new_width, new_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

            // Set texture parameters (example)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        // Update current width, height, and channels for the next layer
        current_width = new_width;
        current_height = new_height;
    }

    // Print intermediate dimensions
    printf("Dimensions after each layer:\n");
    for (int i = 0; i < count; i++) {
        printf("Layer %d: width = %d, height = %d, channels = %d\n",
               i + 1, dimensions[3 * (i + 1)], dimensions[3 * (i + 1) + 1], dimensions[3 * (i + 1) + 2]);
    }


    // Print the final width, height, and channels
    printf("\nFinal dimensions: width = %d, height = %d, channels = %d\n", current_width, current_height, dimensions[3 * count + 2]);

    prep_layers(layers, programbanks, texturebanks, &dimensions, count);

    render_layers(layers, programbanks, texturebanks, &dimensions, count);

    int final_layer_index = count - 1;

    // Get the final texturebank and output dimensions from the last layer
    texturebank *final_texturebank = &texturebanks[final_layer_index + 1]; // final output layer
    int output_width = dimensions[(final_layer_index + 1)*3];  // width for the final output layer
    int output_height = dimensions[(final_layer_index + 1)*3 + 1]; // height for the final output layer

    // Call the function to save the final output texture as a JPEG
    save_final_texture_as_jpeg(final_texturebank, output_width, output_height);

    unsigned char *outputData = (unsigned char*)malloc(output_width * output_height * 4);
    glReadPixels(0, 0, output_width, output_height, GL_RGBA, GL_UNSIGNED_BYTE, outputData);

    // Read back pixels
    /*
    Optional: output the result as a PNG:
        stbi_write_png("output.png", output_width, output_height, 4, outputData, output_width * 4);
    */


    FILE *f1 = fopen(outputfile, "wb");
    if (f1) {
        size_t r1 = fwrite(outputData, sizeof(unsigned char), output_width * output_height * 4, f1);
        fclose(f1);
    }

    free(outputData);

    cleanup_resources(layers, count, programbanks, count, texturebanks, count);

    eglDestroySurface(eglDisplay, eglSurface);
    eglDestroyContext(eglDisplay, eglContext);
    eglTerminate(eglDisplay);

    return 0;
}