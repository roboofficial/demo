% Particle Filter (PF) Localization Simulation
clear; clc;

%% 1. Initialization

dt = 0.1;                 % Time step
t_end = 10;               % Simulation time
t = 0:dt:t_end;

num_particles = 200;      % Number of particles

x_true = [0; 0; 0];       % Ground truth [x; y; theta]

% Initialize particles randomly around the start
particles = randn(3, num_particles) * 0.5;

weights = ones(1, num_particles) / num_particles;

% Noise parameters
Q_std = [0.05; 0.05; 0.02];   % Process noise std dev
R_std = [0.1; 0.1];           % Measurement noise std dev (GPS)

% Preallocate for plotting
history_true = zeros(3, length(t));
history_est  = zeros(3, length(t));

%% 2. Simulation Loop

for i = 1:length(t)

    % --- Control Input ---
    v = 1.0;
    w = 0.1;

    u = [v; w];

    % --- 1. Move the Real Robot (Ground Truth) ---
    x_true = motion_model(x_true, u, dt) + ...
             (Q_std .* randn(3,1));

    % --- 2. Predict Step (Move Particles) ---

    for p = 1:num_particles

        % Move each particle and add noise
        particles(:, p) = motion_model( ...
            particles(:, p), u, dt) + ...
            (Q_std .* randn(3,1));

    end

    % --- 3. Update Step (Weighting based on Measurement) ---

    z = x_true(1:2) + (R_std .* randn(2,1));

    for p = 1:num_particles

        % Distance between particle and measurement
        dist = norm(particles(1:2, p) - z);

        % Calculate Likelihood (Gaussian PDF)
        weights(p) = exp(-(dist^2) / ...
                     (2 * sum(R_std.^2)));

    end

    % Normalize weights
    weights = weights / sum(weights);

    % --- 4. Resampling (Systematic Resampling) ---

    % Prevent particle deprivation
    if 1/sum(weights.^2) < num_particles/2

        cumulative_sum = cumsum(weights);

        for p = 1:num_particles

            sample = rand();

            idx = find(cumulative_sum >= sample, ...
                  1, 'first');

            particles(:, p) = particles(:, idx);

        end

        % Reset weights
        weights = ones(1, num_particles) ...
                  / num_particles;

    end

    % --- State Estimation (Mean of particles) ---
    x_est = mean(particles, 2);

    % Store data
    history_true(:, i) = x_true;
    history_est(:, i)  = x_est;

end

%% 3. Visualization

figure;

plot(history_true(1,:), history_true(2,:), ...
    'g-', 'LineWidth', 2);

hold on;

plot(history_est(1,:), history_est(2,:), ...
    'b--', 'LineWidth', 1.5);

scatter(particles(1,:), particles(2,:), ...
    5, 'r', 'filled', ...
    'MarkerFaceAlpha', 0.2);

legend('True Path', 'PF Estimate', 'Particles');

title('Particle Filter Robot Tracking');

xlabel('X (m)');
ylabel('Y (m)');

grid on;

%% Supporting Motion Model Function

function x_next = motion_model(x, u, dt)

    x_next = x + [u(1) * cos(x(3)) * dt;
                  u(1) * sin(x(3)) * dt;
                  u(2) * dt];

end
