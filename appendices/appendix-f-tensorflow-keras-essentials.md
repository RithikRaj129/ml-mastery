# Appendix F — TensorFlow / Keras Essentials

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics

# ── GPU Setup ─────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Model Definition - Sequential API ────────────────────────────
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# ── Model Definition - Functional API ────────────────────────────
inputs  = keras.Input(shape=(input_dim,))
x       = layers.Dense(128, activation='gelu')(inputs)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model   = keras.Model(inputs=inputs, outputs=outputs)

# ── Compile ───────────────────────────────────────────────────────
model.compile(
    optimizer = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss      = losses.BinaryCrossentropy(),
    metrics   = [metrics.AUC(name='auc'), metrics.BinaryAccuracy(name='acc')]
)

# ── Training ──────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
]

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = 100,
    batch_size      = 64,
    callbacks       = cb_list,
    verbose         = 1
)

# ── Evaluation ────────────────────────────────────────────────────
loss, auc, acc = model.evaluate(X_test, y_test)
y_prob = model.predict(X_test).flatten()

# ── Custom Model with tf.Module ───────────────────────────────────
class CustomModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss   = self.compiled_loss(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# ── Saving ────────────────────────────────────────────────────────
model.save('model.keras')          # Full model
model.save_weights('weights.h5')   # Weights only
loaded = keras.models.load_model('model.keras')

# ── tf.data Pipeline ──────────────────────────────────────────────
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000).batch(64).prefetch(tf.data.AUTOTUNE)

# ── Mixed Precision (faster on GPU) ──────────────────────────────
keras.mixed_precision.set_global_policy('mixed_float16')
```
