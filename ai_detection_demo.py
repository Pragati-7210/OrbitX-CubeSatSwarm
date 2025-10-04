import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------
# STEP 1: Generate Synthetic Dataset
# ------------------------
def generate_star_field(size=64, num_stars=30, asteroid=False, pos=None):
    """Generate synthetic star field with optional asteroid."""
    img = np.zeros((size, size), dtype=np.float32)
    # Random stars
    xs = np.random.randint(0, size, num_stars)
    ys = np.random.randint(0, size, num_stars)
    img[xs, ys] = 1.0  # bright stars
    
    if asteroid and pos is not None:
        img[pos] = 1.0  # asteroid pixel
    
    # Blur stars a little (simulate telescope)
    img = np.clip(img + 0.1 * np.random.randn(size, size), 0, 1)
    return img

def create_dataset(num_samples=2000, size=64):
    X, y = [], []
    for _ in range(num_samples):
        # Half with asteroid, half without
        if np.random.rand() > 0.5:
            pos = (np.random.randint(10, size-10), np.random.randint(10, size-10))
            img = generate_star_field(size=size, asteroid=True, pos=pos)
            y.append(1)  # asteroid
        else:
            img = generate_star_field(size=size, asteroid=False)
            y.append(0)  # no asteroid
        X.append(img)
    return np.array(X)[..., np.newaxis], np.array(y)

print("Generating dataset...")
X, y = create_dataset(2000)
X_train, y_train = X[:1500], y[:1500]
X_test, y_test = X[1500:], y[1500:]

# ------------------------
# STEP 2: Build Tiny CNN Model
# ------------------------
model = keras.Sequential([
    layers.Conv2D(8, (3,3), activation="relu", input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("Training CNN...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# ------------------------
# STEP 3: Simulate Asteroid Moving Across Frames
# ------------------------
fig, ax = plt.subplots()
asteroid_path = [(i, 30+i//3) for i in range(10, 50, 2)]  # diagonal path

for frame, pos in enumerate(asteroid_path):
    img = generate_star_field(size=64, asteroid=True, pos=pos)
    pred = model.predict(img[np.newaxis, ..., np.newaxis], verbose=0)[0][0]

    ax.clear()
    ax.imshow(img, cmap="gray")
    if pred > 0.5:
        rect = patches.Rectangle((pos[1]-2, pos[0]-2), 5, 5, linewidth=1,
                                 edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"Frame {frame}: Asteroid DETECTED ✅ (Prob={pred:.2f})")
    else:
        ax.set_title(f"Frame {frame}: No asteroid ❌ (Prob={pred:.2f})")

    plt.pause(0.5)

plt.show()
