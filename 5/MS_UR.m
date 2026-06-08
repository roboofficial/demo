clc;
clear;

% unicycle
T = 15;
dt = 0.1;
t = 0:dt:T;
N = length(t);
figure;

% Pure Translation
path = getTraj(1,30,0,N,dt);

subplot(1,3,1);
getPlottedPath(path);
title('Pure Translation');

% Pure Rotation
path = getTraj(0,0,1,N,dt);

subplot(1,3,2);
getPlottedPath(path);
title('Pure Rotation');

% Circular Motion
path = getTraj(1,0,1,N,dt);

subplot(1,3,3);
getPlottedPath(path);
title('Circular Motion');



% ================= TRAJECTORY FUNCTION =================
function path = getTraj(v, theta, omega, N, dt)

    theta = deg2rad(theta);

    % x, y, theta
    path = zeros(N,3);

    % Initial Conditions
    path(1,1) = 0;
    path(1,2) = 0;
    path(1,3) = theta;

    for k = 2:N

        % Update x
        path(k,1) = path(k-1,1) + v * dt * cos(theta);

        % Update y
        path(k,2) = path(k-1,2) + v * dt * sin(theta);

        % Update theta
        theta = theta + omega * dt;

        % Store theta
        path(k,3) = theta;

    end
end

% ================= PLOTTING FUNCTION =================
function getPlottedPath(path)

    plot(path(:,1), path(:,2), 'b', 'LineWidth',2);

    hold on;

    % Start Position
    plot(path(1,1), path(1,2), 'go', ...
        'MarkerSize',10,'LineWidth',2);

    % End Position
    plot(path(end,1), path(end,2), 'ro', ...
        'MarkerSize',10,'LineWidth',2);

    % Final orientation
    theta = path(end,3);

    quiver(path(end,1), path(end,2), ...
           cos(theta), sin(theta), ...
           0.5, 'r', 'LineWidth',2);

    grid on;
    axis equal;

    xlabel('X Position');
    ylabel('Y Position');

    title('Robot Trajectory');

    legend('Trajectory','Start','End');

end

% ================= PLOTS =================

