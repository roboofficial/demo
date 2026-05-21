import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
NUM_BOIDS = 50
WIDTH, HEIGHT = 30, 30

MAX_SPEED = 5
NEIGHBOR_RADIUS = 1.5
SEPARATION_DISTANCE = 1

# Weights (tune these!)
W_ALIGNMENT = 1.0
W_COHESION = 0.8
W_SEPARATION = 1.2

# -----------------------------
# Initialize boids
# -----------------------------
positions = np.random.rand(NUM_BOIDS, 2) * [WIDTH, HEIGHT]
velocities = (np.random.rand(NUM_BOIDS, 2) - 0.5) * 0.2

# -----------------------------
# Helper functions
# -----------------------------
def limit_speed(v):
    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        return (v / speed) * MAX_SPEED
    return v

# -----------------------------
# Boid update rules
# -----------------------------
def update_boids():
    global positions, velocities
    new_velocities = np.copy(velocities)

    for i in range(NUM_BOIDS):
        pos_i = positions[i]
        vel_i = velocities[i]

        neighbors = []
        for j in range(NUM_BOIDS):
            if i != j:
                dist = np.linalg.norm(positions[j] - pos_i)
                if dist < NEIGHBOR_RADIUS:
                    neighbors.append(j)

        if not neighbors:
            continue

        # -----------------------------
        # Alignment
        # -----------------------------
        avg_vel = np.mean(velocities[neighbors], axis=0)
        alignment = avg_vel - vel_i

        # -----------------------------
        # Cohesion
        # -----------------------------
        center = np.mean(positions[neighbors], axis=0)
        cohesion = center - pos_i

        # -----------------------------
        # Separation
        # -----------------------------
        separation = np.zeros(2)
        for j in neighbors:
            diff = pos_i - positions[j]
            dist = np.linalg.norm(diff)
            if dist < SEPARATION_DISTANCE and dist > 0:
                separation += diff / dist

        # Combine forces
        acceleration = (
            W_ALIGNMENT * alignment +
            W_COHESION * cohesion +
            W_SEPARATION * separation
        )

        new_velocities[i] += acceleration * 0.01
        new_velocities[i] = limit_speed(new_velocities[i])

    velocities = new_velocities
    positions += velocities

    # Wrap around boundaries
    positions[:, 0] %= WIDTH
    positions[:, 1] %= HEIGHT

# -----------------------------
# Plot animation
# -----------------------------
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], marker='.')

ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title("Boids Flocking (Reynolds Model)")

def animate(frame):
    update_boids()
    scat.set_offsets(positions)
    return scat,

ani = FuncAnimation(fig, animate, interval=50)

plt.show()