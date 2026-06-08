clc;
clear;

dt=0.1;
T=15;
t = 0:dt:T;
N = length(t);

v_d = 0.5;

Q = diag([10,10,5]);
R = diag([1,0.5]);

A = [0,0,0;
    0,0,v_d;
    0,0,0];
B=[-1,0; 0, 0; 0,-1];

Goal = [3,3,deg2rad(30)];
X = zeros(N,3); % x,y,theta
e_t = zeros(N,1);

Co = ctrb(A,B);
fprintf('Controllability rank: %d / %d\n', rank(Co), size(A,1));

[K, ~, ~] = lqr(A,B,Q,R);

fprintf('LQR Gain = \n'); disp(K);

C = zeros(N,2);

for k=2:N
    e_x = (Goal(1) - X(k-1,1)) * cos(X(k-1,3)) + (Goal(2) - X(k-1,2)) * sin(X(k-1,3));
    e_y = - (Goal(1) - X(k-1,1)) * sin(X(k-1,3)) + (Goal(2) - X(k-1,2)) * cos(X(k-1,3));
    e_z = atan2(sin(Goal(3) - X(k-1,3)), cos(Goal(3) - X(k-1,3)));

    e = [e_x; e_y; e_z];
    u = - K * e;
    v = v_d + u(1); % linear velocity
 w = u(2); % angular velocity
 % Saturate inputs
 v = max(-1.0, min(1.0, v));
 w = max(-2.0, min(2.0, w));
   C(k,:) = [v, w];
 dX = [dt * v * cos(X(k-1,3));
     dt * v * sin(X(k-1,3));
     dt *w];
 X(k,:) = X(k-1,:) + dX';
 X(k,3) = atan2(sin(X(k,3)), cos(X(k,3)));
 e_t(k,1) = sqrt(e_x^2 + e_y^2);
end


fprintf('Final position error: %.4f m\n', e_t(end,1));


%% ===== PART e: PLOTS =====
% --- 1. Robot Path ---
figure('Color','w');
plot(X(:,1), X(:,2), 'b-', 'LineWidth', 2); hold on;
plot(X(1,1), X(1,2), 'go', 'MarkerSize', 10, 'MarkerFaceColor','g');
plot(Goal(1), Goal(2), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('X (m)'); ylabel('Y (m)');
title('LQR - Robot Path');
legend('Path', 'Start', 'Goal');
grid on; axis equal;

% --- 2. Position Error ---
figure('Color','w');
plot(t, e_t(:,1), 'b-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Position Error (m)');
title('LQR - Position Error vs Time');
grid on;

% --- 3. Control Inputs ---
figure('Color','w');
subplot(2,1,1);
plot(t, C(:,1), 'b-', 'LineWidth', 2);
ylabel('v (m/s)'); title('LQR - Linear Velocity');
grid on;
subplot(2,1,2);
plot(t, C(:,2), 'r-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('\omega (rad/s)');
title('LQR - Angular Velocity');
grid on;
% --- 4. X, Y, Theta over time ---
figure('Color','w');
subplot(3,1,1);
plot(t, X(:,1), 'b', 'LineWidth', 1.5); yline(Goal(1),'r--');
ylabel('X (m)'); title('LQR - State Trajectories'); grid on;
subplot(3,1,2);
plot(t, X(:,2), 'b', 'LineWidth', 1.5); yline(Goal(2),'r--');
ylabel('Y (m)'); grid on;
subplot(3,1,3);
plot(t, X(:,3), 'b', 'LineWidth', 1.5); yline(Goal(3),'r--');
ylabel('\theta (rad)'); xlabel('Time (s)'); grid on;
