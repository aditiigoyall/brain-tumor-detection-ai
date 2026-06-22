import tensorflow as tf
import numpy as np
import cv2

LAST_CONV_LAYER = "top_conv"


def build_gradcam_model(model):
    """
    Builds the Grad-CAM model for EfficientNetB0.
    Returns the Grad-CAM sub-model and classifier head.
    """

    base_model = model.get_layer("efficientnetb0")

    gradcam_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(LAST_CONV_LAYER).output,
            base_model.output,
        ],
    )

    head_layers = model.layers[2:]

    return gradcam_model, head_layers


def make_gradcam_heatmap(img_tensor, gradcam_model, head_layers, pred_index=None):
    """
    Generates a Grad-CAM heatmap.

    Args:
        img_tensor: Preprocessed input image of shape (1, 224, 224, 3)
        gradcam_model: Grad-CAM model returned by build_gradcam_model()
        head_layers: Classification head layers
        pred_index: Target class index (optional)

    Returns:
        Normalized Grad-CAM heatmap.
    """

    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:

        conv_outputs, base_output = gradcam_model(
            img_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = base_output

        for layer in head_layers:
            x = layer(x, training=False)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.45):
    """
    Overlays the Grad-CAM heatmap on the original image.

    Args:
        image: Original image (BGR format)
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor

    Returns:
        RGB image with heatmap overlay.
    """

    height, width = image.shape[:2]

    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    )
