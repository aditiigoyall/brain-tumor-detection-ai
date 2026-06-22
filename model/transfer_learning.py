from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(train_y):

    train_labels = np.argmax(train_y, axis=1)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    return dict(enumerate(class_weights))
