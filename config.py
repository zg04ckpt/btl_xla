# Configuration settings for the Handwritten Digit Recognition application

class Config:
    # Path to the trained models
    DIGIT_MODEL_PATH = 'train/mnist_cnn_model.h5'
    SHAPE_MODEL_PATH = 'train/shapes_cnn_model_best.h5'
    
    # Image processing settings
    DIGIT_IMAGE_SIZE = 28
    SHAPE_IMAGE_SIZE = 64
    IMAGE_CHANNELS = 1  # Grayscale images
    
    # Recognition classes
    DIGIT_CLASSES = list(range(10))  # 0-9
    SHAPE_CLASSES = ['circle', 'rectangle', 'triangle']

    # Preprocessing settings
    NOISE_REMOVAL_KERNEL_SIZE = 5
    SEGMENTATION_THRESHOLD = 0.5

    # GUI settings
    WINDOW_TITLE = "Handwritten Recognition - Digits & Shapes"
    WINDOW_SIZE = (800, 600)

    # Confidence score threshold for displaying results
    CONFIDENCE_THRESHOLD = 0.7

    # Debugging settings
    DEBUG_MODE = True
    DEBUG_IMAGES_PATH = 'debug_images/'