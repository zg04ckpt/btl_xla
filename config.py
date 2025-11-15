# Configuration settings for the Handwritten Digit Recognition application

class Config:
    # Path to the trained model
    MODEL_PATH = 'models/mnist_cnn_model.h5'
    
    # Image processing settings
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    IMAGE_CHANNELS = 1  # Grayscale images

    # Preprocessing settings
    NOISE_REMOVAL_KERNEL_SIZE = 5
    SEGMENTATION_THRESHOLD = 0.5

    # GUI settings
    WINDOW_TITLE = "Handwritten Digit Recognition"
    WINDOW_SIZE = (800, 600)

    # Confidence score threshold for displaying results
    CONFIDENCE_THRESHOLD = 0.7

    # Debugging settings
    DEBUG_MODE = True
    DEBUG_IMAGES_PATH = 'debug_images/'