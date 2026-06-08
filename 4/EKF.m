
% EKF Localization/Tracking Simulation
clear; clc;

%% 1. Initialization
dt = 0.1;                 % Time step
t_end = 10;               % Simulation time
t = 0:dt:t_end;

% State: [x; y; theta]
x_true = [0; 0; 0];       % Ground truth
x_est = [0; 0; 0];        % EKF estimate
P = eye(3) * 0.1;         % Covariance matrix

% Noise Matrices
Q = diag([0.01, 0.01, 0.005]); % Process noise (motion)
R = diag([0.1, 0.1]);          % Measurement noise (GPS-like)

% Preallocate for plotting
history_true = zeros(3, length(t));
history_est  = zeros(3, length(t));

%% 2. Simulation Loop
for i = 1:length(t)

    % --- Control Input (v = velocity, w = angular velocity) ---
    v = 1.0;
    w = 0.1;

    u = [v; w];

    % --- Real System (Physics) ---
    x_true = motion_model(x_true, u, dt) + sqrt(Q) * randn(3,1);

    % --- EKF Step 1: Prediction ---
    x_pred = motion_model(x_est, u, dt);

    % Jacobian of motion model (F)
    F = [1, 0, -v*sin(x_est(3))*dt;
         0, 1,  v*cos(x_est(3))*dt;
         0, 0,  1];

    P_pred = F * P * F' + Q;

    % --- EKF Step 2: Update (Measurement) ---
    % Assume we observe [x; y] position (GPS)

    z = x_true(1:2) + sqrt(R) * randn(2,1);

    H = [1, 0, 0;
         0, 1, 0];   % Observation Jacobian

    % Kalman Gain
    K = P_pred * H' / (H * P_pred * H' + R);

    % State Update
    x_est = x_pred + K * (z - H * x_pred);

    % Covariance Update
    P = (eye(3) - K * H) * P_pred;

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
    'r--', 'LineWidth', 1.5);

legend('True Path', 'EKF Estimate');

title('EKF Robot Localization');

xlabel('X (m)');
ylabel('Y (m)');

grid on;

%% Supporting Motion Model Function
function x_next = motion_model(x, u, dt)

    % Simple Unicycle Model

    x_next = x + [u(1) * cos(x(3)) * dt;
                  u(1) * sin(x(3)) * dt;
                  u(2) * dt];
end
