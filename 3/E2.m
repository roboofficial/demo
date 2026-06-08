%% Sensor Fusion Lab - Kalman Filter Example
% Simulate vehicle position estimation using GPS and IMU
clear all; close all; clc;

%% System Parameters
dt = 0.1; % Time step (100 ms)
total_time = 30; % Total simulation time (seconds)
t = 0:dt:total_time; % Time vector
N = length(t); % Number of samples

% True vehicle motion (ground truth)
true_acceleration = 0.5; % Constant acceleration (m/s^2)
true_velocity = zeros(1,N);
true_position = zeros(1,N);

% Generate true trajectory
for k = 2:N
    true_velocity(k) = true_velocity(k-1) + true_acceleration * dt;
    true_position(k) = true_position(k-1) + true_velocity(k-1) * dt + 0.5 * true_acceleration * dt^2;
end

%% Sensor Noise Parameters
gps_noise_std = 10; % GPS position noise (meters)
imu_noise_std = 0.2; % IMU acceleration noise (m/s^2)
% Generate noisy measurements
gps_measurements = true_position + gps_noise_std * randn(1,N);
imu_measurements = true_acceleration + imu_noise_std * randn(1,N);

% Visualize raw sensor data
figure('Name','Raw Sensor Data');

subplot(2,1,1);
plot(t, true_position, 'g-', 'LineWidth', 2); hold on;
plot(t, gps_measurements, 'b.', 'MarkerSize', 3);
xlabel('Time (s)');
ylabel('Position (m)');
legend('True Position', 'GPS Measurements');
title('GPS Sensor Data');
grid on;

subplot(2,1,2);
plot(t, true_acceleration * ones(1,N), 'g-', 'LineWidth', 2); hold on;
plot(t, imu_measurements, 'r.', 'MarkerSize', 3);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
legend('True Acceleration', 'IMU Measurements');
title('IMU Sensor Data');
grid on;

%% Kalman Filter
%% Kalman Filter Initialization
% State vector: [position; velocity; acceleration]
x = [0; 0; 0]; % Initial state estimate
% State covariance matrix
P = eye(3) * 100; % Initial uncertainty
% State transition matrix (constant acceleration model)
F = [1 dt 0.5*dt^2;
0 1 dt;
0 0 1];
% Process noise covariance
q = 0.2; % Process noise intensity
Q = q * [dt^4/4 dt^3/2 dt^2/2;
dt^3/2 dt^2 dt;
dt^2/2 dt 1];
% Measurement noise covariance
R_gps = gps_noise_std^2; % GPS variance
R_imu = imu_noise_std^2; % IMU variance

%% Kalman Filter Execution
% Storage for results
x_est = zeros(3, N); % Estimated states
x_est(:,1) = x;
for k = 2:N

%% Prediction Step
    x_pred = F * x;
    P_pred = F * P * F' + Q;

%% Update Step - GPS Measurement
    H_gps = [1 0 0]; % GPS measures position only
    z_gps = gps_measurements(k);
% Innovation
    y_gps = z_gps - H_gps * x_pred;
% Innovation covariance
    S_gps = H_gps * P_pred * H_gps' + R_gps;
% Kalman gain
    K_gps = P_pred * H_gps' / S_gps;
% State update with GPS
    x = x_pred + K_gps * y_gps;
    P = (eye(3) - K_gps * H_gps) * P_pred;

%% Update Step - IMU Measurement
    H_imu = [0 0 1]; % IMU measures acceleration
    z_imu = imu_measurements(k);

% Innovation
    y_imu = z_imu - H_imu * x;

% Innovation covariance
    S_imu = H_imu * P * H_imu' + R_imu;

% Kalman gain
    K_imu = P * H_imu' / S_imu;

% State update with IMU
    x = x + K_imu * y_imu;
    P = (eye(3) - K_imu * H_imu) * P;

% Store results
    x_est(:,k) = x;
end

%% Results Visualization
figure('Name','Kalman Filter Results');
% Position estimation
subplot(3,1,1);
plot(t, true_position, 'g-', 'LineWidth', 2); hold on;
plot(t, gps_measurements, 'b.', 'MarkerSize', 4);
plot(t, x_est(1,:), 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Position (m)');
legend('True Position', 'GPS Measurements', 'Kalman Filter Estimate');
title('Position Estimation');
grid on;

% Velocity estimation
subplot(3,1,2);
plot(t, true_velocity, 'g-', 'LineWidth', 2); hold on;
plot(t, x_est(2,:), 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Velocity (m/s)');
legend('True Velocity', 'Kalman Filter Estimate');
title('Velocity Estimation');
grid on;

% Acceleration estimation
subplot(3,1,3);
plot(t, true_acceleration*ones(1,N), 'g-', 'LineWidth', 2); hold on;
plot(t, imu_measurements, 'b.', 'MarkerSize', 4);
plot(t, x_est(3,:), 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
legend('True Acceleration', 'IMU Measurements', 'Kalman Filter Estimate');
title('Acceleration Estimation');
grid on;

%% Performance Analysis
% Root Mean Square Error (RMSE)
rmse_position = sqrt(mean((true_position - x_est(1,:)).^2));
rmse_velocity = sqrt(mean((true_velocity - x_est(2,:)).^2));
rmse_acceleration = sqrt(mean((true_acceleration - x_est(3,:)).^2));

% Display results
fprintf('\n=== Sensor Fusion Performance ===\n');
fprintf('Position RMSE: %.3f meters\n', rmse_position);
fprintf('Velocity RMSE: %.3f m/s\n', rmse_velocity);
fprintf('Acceleration RMSE: %.3f m/s^2\n', rmse_acceleration);

% Improvement over GPS alone
gps_rmse = sqrt(mean((true_position - gps_measurements).^2));
improvement = ((gps_rmse - rmse_position) / gps_rmse) * 100;
fprintf('Improvement over GPS: %.2f%%\n', improvement);

%% Real-time Performance Metrics
% Measure execution time
tic;
for k = 2:1000
x_temp = F * x;
P_temp = F * P * F' + Q;
K_temp = P_temp * H_gps' / (H_gps * P_temp * H_gps' + R_gps);
x = x_temp + K_temp * (0 - H_gps * x_temp);
end
elapsed_time = toc;
processing_time = elapsed_time / 998; % Average per iteration
fps = 1 / processing_time;
fprintf('\n=== Real-time Performance ===\n');
fprintf('Average processing time: %.6f ms\n', processing_time*1000);
fprintf('Achievable frame rate: %.1f Hz\n', fps);
