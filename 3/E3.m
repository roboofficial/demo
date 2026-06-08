clc;
clear;
close all;

%% Robot Parameters
L1 = 1;
L2 = 0.8;

%% Simulation Parameters
dt = 0.01;
t = 0:dt:10;
N = length(t);

%% Desired Circular Path
r = 0.4;
xc = 1;
yc = 0.5;

x_des = xc + r*cos(t);
y_des = yc + r*sin(t);

%% Inverse Kinematics
theta1_des = zeros(1,N);
theta2_des = zeros(1,N);

for i = 1:N

    x = x_des(i);
    y = y_des(i);

    D = (x^2 + y^2 - L1^2 - L2^2)/(2*L1*L2);
    

    theta2_des(i) = atan2(sqrt(1-D^2),D);

    theta1_des(i) = atan2(y,x) - atan2(L2*sin(theta2_des(i)),...
                      L1 + L2*cos(theta2_des(i)));
end

%% Velocity and Acceleration
theta1_dot_des = gradient(theta1_des,dt);
theta2_dot_des = gradient(theta2_des,dt);

theta1_ddot_des = gradient(theta1_dot_des,dt);
theta2_ddot_des = gradient(theta2_dot_des,dt);

%% PID Gains
Kp = 100;
Kd = 20;

%% Initial Conditions
theta1 = 0;
theta2 = 0;

theta1_dot = 0;
theta2_dot = 0;

%% Storage
theta1_actual = zeros(1,N);
theta2_actual = zeros(1,N);

tau1_applied = zeros(1,N);
tau2_applied = zeros(1,N);

x_actual = zeros(1,N);
y_actual = zeros(1,N);

%% Simulation Loop
for i = 1:N

    %% Errors
    e1 = theta1_des(i) - theta1;
    e2 = theta2_des(i) - theta2;

    e1_dot = theta1_dot_des(i) - theta1_dot;
    e2_dot = theta2_dot_des(i) - theta2_dot;

    %% Desired Torque
    tau1_desired = theta1_ddot_des(i);
    tau2_desired = theta2_ddot_des(i);

    %% Applied Torque using PID
    tau1 = tau1_desired + Kp*e1 + Kd*e1_dot;
    tau2 = tau2_desired + Kp*e2 + Kd*e2_dot;

    %% Robot Dynamics
    theta1_ddot = tau1;
    theta2_ddot = tau2;

    %% Update Velocity
    theta1_dot = theta1_dot + theta1_ddot*dt;
    theta2_dot = theta2_dot + theta2_ddot*dt;

    %% Update Position
    theta1 = theta1 + theta1_dot*dt;
    theta2 = theta2 + theta2_dot*dt;

    %% Store Values
    theta1_actual(i) = theta1;
    theta2_actual(i) = theta2;

    tau1_applied(i) = tau1;
    tau2_applied(i) = tau2;

    %% Actual End Effector Position
    x_actual(i) = L1*cos(theta1) + ...
                  L2*cos(theta1 + theta2);

    y_actual(i) = L1*sin(theta1) + ...
                  L2*sin(theta1 + theta2);

end

%% PLOT 1 : Desired vs Actual Path
figure;

plot(x_des,y_des,'r--','LineWidth',3);
hold on;

plot(x_actual,y_actual,'b','LineWidth',2);

xlabel('X');
ylabel('Y');

title('End Effector Path Tracking');

legend('Desired Path','Actual Path');

axis equal;
grid on;

%% PLOT 2 : Joint Angles
figure;

subplot(2,1,1)

plot(t,rad2deg(theta1_des),'r--','LineWidth',3);
hold on;

plot(t,rad2deg(theta1_actual),'b','LineWidth',2);

xlabel('Time');
ylabel('Theta1 (deg)');

legend('Desired','Actual');

title('Joint 1');

grid on;

subplot(2,1,2)

plot(t,rad2deg(theta2_des),'r--','LineWidth',3);
hold on;

plot(t,rad2deg(theta2_actual),'b','LineWidth',2);

xlabel('Time');
ylabel('Theta2 (deg)');

legend('Desired','Actual');

title('Joint 2');

grid on;

%% PLOT 3 : Applied Torque
figure;

subplot(2,1,1)

plot(t,tau1_applied,'LineWidth',2);

xlabel('Time');
ylabel('Torque 1');

title('Applied Torque Joint 1');

grid on;

subplot(2,1,2)

plot(t,tau2_applied,'LineWidth',2);

xlabel('Time');
ylabel('Torque 2');

title('Applied Torque Joint 2');

grid on;


%% PLOT 4 : 3D Trajectory with Time

figure;

plot3(x_des, y_des, t, 'r--', 'LineWidth',2);
hold on;

plot3(x_actual, y_actual, t, 'b', 'LineWidth',2);

xlabel('X Position');
ylabel('Y Position');
zlabel('Time');

title('3D End Effector Trajectory');

legend('Desired Trajectory','Actual Trajectory');

grid on;
view(3);
